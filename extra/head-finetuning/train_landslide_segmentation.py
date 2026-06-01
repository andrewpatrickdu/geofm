#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that trains a landslide detector

python train_landslide_segmentation.py --batch_size 20 --num_epochs 100 --encoder /data/ckpt-mae-256/model-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-landslide-segmentation-mae-256

python train_landslide_segmentation.py --batch_size 20 --num_epochs 100 --encoder /data/ckpt-mae-512/model-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-landslide-segmentation-mae-512

python train_landslide_segmentation.py --batch_size 20 --num_epochs 100 --encoder /data/ckpt-distillation-256/model-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-landslide-segmentation-distillation-256

python train_landslide_segmentation.py --batch_size 20 --num_epochs 100 --encoder /data/ckpt-distillation-512/model-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-landslide-segmentation-distillation-512

python train_landslide_segmentation.py --batch_size 20 --num_epochs 100 --encoder /data/Prithvi_EO_V2_300M.pt --encoder_embed_dim 1024 --checkpoint /data/ckpt-landslide-segmentation-baseline-1024

"""

###############################################################################
import logging
import os
import random
import timeit
import time
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from prithvi_mae import PrithviViT

name_classes = np.array(['non-landslide','landslide'], dtype=str)
epsilon = 1e-14

# set random seeds for reproducibility
seed = 0
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
torch.cuda.manual_seed(seed) 
np.random.seed(0) 
random.seed(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parse command-line arguments
import argparse
parser = argparse.ArgumentParser(description='DUA')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--encoder', type=str, default='/home/andrew/GFM/ckpt-mae-256/model-final.pt', help='Directory of encoder pretrained weights')
parser.add_argument('--encoder_embed_dim', type=int, default=256, help='Encoder embedding dimension')
parser.add_argument('--checkpoint', type=str, default='/home/andrew/GFM/ckpt-cloud-mae-256', help='Directory of training and evaluation results')
args = parser.parse_args()

filepath = args.checkpoint
if not os.path.exists(filepath):
    os.makedirs(f'{filepath}')

# Set up logging
log_filename = f"{filepath}/training.log"
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    # format='%(asctime)s - %(levelname)s - %(message)s',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

# Log the start of the script
logging.info("Script started.")

# Set the device to GPU ("cuda") if available; otherwise, default to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

###############################################################################
# PYTORCH DATA LOADER
###############################################################################
import h5py

class LandslideDataSet(Dataset):
    def __init__(self, data_dir, list_path, max_iters=None, set='labeled'):
        self.list_path = list_path
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
           
        if not max_iters==None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]

        self.files = []

        if set=='labeled':
            for name in self.img_ids:
                img_file = data_dir + name
                label_file = data_dir + name.replace('img','mask').replace('image','mask')
                self.files.append({
                    'img': img_file,
                    'label': label_file,
                    'name': name
                })
        elif set=='unlabeled':
            for name in self.img_ids:
                img_file = data_dir + name
                self.files.append({
                    'img': img_file,
                    'name': name
                })
            
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        
        if self.set=='labeled':
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            with h5py.File(datafiles['label'], 'r') as hf:
                label = hf['mask'][:]
            name = datafiles['name']
                
            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]

            # # Min-max normalization per channel
            # min_vals = image.min(axis=(1, 2), keepdims=True)
            # max_vals = image.max(axis=(1, 2), keepdims=True)
            # image = (image - min_vals) / (max_vals - min_vals + 1e-8)

            return image.copy(), label.copy(), np.array(size), name

        else:
            with h5py.File(datafiles['img'], 'r') as hf:
                image = hf['img'][:]
            name = datafiles['name']
                
            image = np.asarray(image, np.float32)
            image = image.transpose((-1, 0, 1))
            size = image.shape

            for i in range(len(self.mean)):
                image[i,:,:] -= self.mean[i]
                image[i,:,:] /= self.std[i]
            
            # # Min-max normalization per channel
            # min_vals = image.min(axis=(1, 2), keepdims=True)
            # max_vals = image.max(axis=(1, 2), keepdims=True)
            # image = (image - min_vals) / (max_vals - min_vals + 1e-8)

            return image.copy(), np.array(size), name

data_dir ='/data/landslide-detection/'

training_dir = '/data/landslide-detection/train.txt'
validation_dir = '/data/landslide-detection/valid.txt'
testing_dir = '/data/landslide-detection/test.txt'

train_dataset = LandslideDataSet(data_dir=data_dir, list_path=training_dir)
val_dataset = LandslideDataSet(data_dir=data_dir, list_path=validation_dir)
test_dataset = LandslideDataSet(data_dir=data_dir, list_path=testing_dir)

batch_size = args.batch_size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
logging.info(f"Loaded training dataset with {len(train_dataset)} samples.")
logging.info(f"Loaded validation dataset with {len(val_dataset)} samples.")
logging.info(f"Loaded testing dataset with {len(test_dataset)} samples.")


# # CHECKER: total number of landslide and not landslide pixels in dataset
# landslide_pixels = 0
# not_landslide_pixels = 0
# missing_data = 0
# for _, mask, _, _ in test_loader:

#     mask = mask.numpy()

#     landslide_pixels += np.sum(mask == 1)
#     not_landslide_pixels += np.sum(mask == 0)
#     missing_data += np.sum(mask == -1)
# print(f"- Total landslide pixels: {landslide_pixels}")
# print(f"- Total not landslide pixels: {not_landslide_pixels}")
# print(f"- Total missing value pixels: {missing_data}")
# print(f"- Total pixels: {landslide_pixels + not_landslide_pixels + missing_data}")

'''
Training: 
- Total landslide pixels: 1,442,790 (2.3%)
- Total not landslide pixels: 60,800,026 (97.7%)
- Total missing value pixels: 0
- Total pixels: 62,242,816


Validation
- Total landslide pixels: 68,745 (1.7%)
- Total not landslide pixels: 3,945,335 (98.3)
- Total missing value pixels: 0
- Total pixels: 4,014,080


Testing
- Total landslide pixels: 247,531 (1.9%)
- Total not landslide pixels: 12,859,669 (98.1%)
- Total missing value pixels: 0
- Total pixels: 13,107,200
'''

###############################################################################
# ENCODER MODEL
###############################################################################
# Initialize model
encoder = PrithviViT(img_size=224, embed_dim = args.encoder_embed_dim).to(device)
logging.info("Encoder initialized.")

# CHCKER: Display total number of parameters
total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
logging.info(f"Encoder has {total_params:,} parameters.")

# Load checkpoint (pretrained weights)
checkpoint = args.encoder
state_dict = torch.load(checkpoint, map_location=device)

# Discard fixed pos_embedding weight
for k in list(state_dict.keys()):
    if 'pos_embed' in k:
        del state_dict[k]
        
encoder.load_state_dict(state_dict, strict=False)
logging.info(f"Loaded checkpoint from {checkpoint}.")

# Freeze parameters    
for name, param in encoder.named_parameters():
    param.requires_grad = False   

# # CHECKER - display trainable model parameters "
# for name, param in encoder.named_parameters():
#     if param.requires_grad == True:
#         # print(name, param.data)
#         logging.info(name)

###############################################################################
# SEGMENTATION DECODER MODEL
###############################################################################
import torch.nn as nn

# UPerNet adaptation for ViT-style encoders
class ConvModule(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels // 2
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(in_channels, out_channels, kernel_size=1, padding=0),
            ) for scale in pool_sizes
        ])
        total_channels = in_channels + len(pool_sizes) * out_channels
        self.bottleneck = ConvModule(total_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h, w = x.shape[2:]
        pooled = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        x = torch.cat([x] + pooled, dim=1)
        return self.bottleneck(x)

class UperNetDecoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2, ppm_pool_scales=(1, 2, 3, 6)):
        super().__init__()

        reduced_channels = 80 # 256  

        self.lateral_convs = nn.ModuleList([
            ConvModule(input_dim, reduced_channels, kernel_size=1, padding=0) for _ in range(4)
        ])

        self.ppm = PyramidPoolingModule(reduced_channels, pool_sizes=ppm_pool_scales, out_channels=reduced_channels // 2)

        self.fpn_convs = nn.ModuleList([
            ConvModule(reduced_channels, reduced_channels) for _ in range(4)
        ])

        self.fpn_bottleneck = ConvModule(reduced_channels * 4, reduced_channels)

        self.classifier = nn.Conv2d(reduced_channels, num_classes, kernel_size=1)

    def forward(self, reshaped_feats: list[torch.Tensor]):
        assert len(reshaped_feats) == 4, "UperNetDecoder expects 4 input features"

        laterals = []
        for feat, conv in zip(reshaped_feats, self.lateral_convs):
            B, N, D = feat.shape
            H = W = int(N ** 0.5)
            feat = feat.permute(0, 2, 1).reshape(B, D, H, W)
            laterals.append(conv(feat))

        laterals[-1] = self.ppm(laterals[-1])

        for i in range(3, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=False)
            laterals[i - 1] = laterals[i - 1] + upsampled

        fpn_outs = [fpn_conv(lat) for fpn_conv, lat in zip(self.fpn_convs, laterals)]
        target_size = fpn_outs[0].shape[2:]
        fpn_outs = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in fpn_outs]

        x = torch.cat(fpn_outs, dim=1)
        x = self.fpn_bottleneck(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.classifier(x)

classifier = UperNetDecoder(input_dim=encoder.embed_dim, num_classes=2).to(device)

###############################################################################

logging.info("Decoder initialized.")

# CHCKER: Display total number of parameters
total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
logging.info(f"Decoder has {total_params:,} parameters.")

###############################################################################
# TRAINING PARAMETERS
###############################################################################  
# Define loss
weight = torch.tensor([1., 4.]).cuda() # default
criterion = nn.CrossEntropyLoss(
    weight=weight, size_average=None, ignore_index=255, reduce=None, reduction='mean')

# criterion = nn.CrossEntropyLoss(size_average=None, ignore_index=255, reduce=None, reduction='mean')

# Define optimiser
optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-5, weight_decay=0.1)

# Define learning rate
num_epochs = args.num_epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(    
    optimizer,
    T_0=len(train_loader) * 10,
    T_mult=2,
    eta_min=1e-6,
    last_epoch=-1
)

###############################################################################
# FINE-TUNING
############################################################################### 
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class=19):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def eval_image(predict,label,num_classes):
    index = np.where((label>=0) & (label<num_classes))
    predict = predict[index]
    label = label[index] 
    
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    for i in range(0,num_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)        
    
    return TP,FP,TN,FN,len(label)

# Start timer
start = timeit.default_timer()
logging.info("Training started.")
logging.info(f"Training Parameters - Batch Size: {batch_size}, Epochs: {num_epochs}, Encoder embed_dim: {encoder.embed_dim}")
 
# Perform fine-tuning
F1_best = 0.
mIoU_best = 0.
for epoch in range(1, num_epochs + 1):
    
    # start timer
    t0 = time.time()

    hist = np.zeros((len(train_loader), 3))
    
    encoder.eval()
    classifier.train()
    for batch_index, data in enumerate(train_loader):   
 
        # load images, labels and random noise to device (GPU or CPU) 
        images, labels, size, name = data
        images = images[:,[1,2,3,8,10,11],:,:].to(device, dtype=torch.float)  
        labels = labels.unsqueeze(1).to(device)        
        
        # resize images and labels
        images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
        labels_cropped = labels_cropped.squeeze(1)

        # Apply same random flipping
        batch_size = images_cropped.shape[0]
        for i in range(batch_size):
            if torch.rand(1) < 0.5:  # Random horizontal flip
                images_cropped[i] = transforms.functional.hflip(images_cropped[i])
                labels_cropped[i] = transforms.functional.hflip(labels_cropped[i])
    
            if torch.rand(1) < 0.5:  # Random vertical flip
                images_cropped[i] = transforms.functional.vflip(images_cropped[i])
                labels_cropped[i] = transforms.functional.vflip(labels_cropped[i])

        # # CHECKER - plot image
        # VIZ_FACTOR = 1 #2.5
        # idx = 17
        # img = images[idx].permute(1, 2, 0).cpu().numpy() * VIZ_FACTOR
        # mask = labels[idx].cpu().numpy()
        # mask = mask.squeeze(0)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(img[:, :, [2, 1, 0]])  # Assuming BGR, swap to RGB
        # axes[0].set_title("Image")
        # # axes[0].axis("off")
        # axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)  # Ensure correct scaling
        # axes[1].set_title("Mask")
        # # axes[1].axis("off")
        # plt.tight_layout()
        # plt.show()

        # # CHECKER - plot cropped image
        # img = images_cropped[idx].permute(1, 2, 0).cpu().numpy() * VIZ_FACTOR
        # mask = labels_cropped[idx].cpu().numpy()
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(img[:, :, [2, 1, 0]])  # Assuming BGR, swap to RGB
        # axes[0].set_title("Image")
        # # axes[0].axis("off")
        # axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)  # Ensure correct scaling
        # axes[1].set_title("Mask")
        # # axes[1].axis("off")
        # plt.tight_layout()
        # plt.show()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        #######################################################################
        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped.unsqueeze(2)) 
        # print(len(features), features[0].shape)
        
        selected_feats = [features[i] for i in [5, 11, 17, 23]]
        # print(len(selected_feats), selected_feats[0].shape)
                
        # remove CLS token before passing to segmentation head
        selected_feats = [f[:, 1:, :] for f in selected_feats]
        # print(len(selected_feats), selected_feats[0].shape)
        
        # # reshape features for use of segmentation head
        # reshaped_feats = []
        # for f in selected_feats:
        #     B, N, D = f.shape
        #     H = W = int(N ** 0.5)  # assumes square patch grid
        #     f = f.permute(0, 2, 1).reshape(B, D, H, W)
        #     reshaped_feats.append(f)
        # # print(len(reshaped_feats), reshaped_feats[0].shape)
        
        # logits = classifier(reshaped_feats)

        # compute logits
        logits = classifier(selected_feats)
        
        # compute batch loss
        loss = criterion(logits, labels_cropped)
        
        # compute metrics - mIoU, overall accruacy and loss
        _, predicted = torch.max(logits, 1)
        lbl_pred = predicted.detach().cpu().numpy()
        lbl_true = labels_cropped.detach().cpu().numpy()
        metrics_batch = []
        for lt, lp in zip(lbl_true, lbl_pred):
            _,_,mean_iu,_ = label_accuracy_score(lt, lp, n_class=2)
            metrics_batch.append(mean_iu)                
        batch_miou = np.nanmean(metrics_batch, axis=0)  
        batch_oa = np.sum(lbl_pred==lbl_true)*1./len(lbl_true.reshape(-1))        
        
        hist[batch_index,0] = loss.item()
        hist[batch_index,1] = batch_oa
        hist[batch_index,2] = batch_miou        

        # backward propagation: compute gradient of the loss wrt model parameters
        loss.backward()

        # update the model parameters
        optimizer.step()
        
        # update learning rate
        scheduler.step(epoch + batch_index / len(train_loader)) 

    logging.info('Epoch: %d Iter: %d/%d Time: %.2f OA: %.2f mIoU: %.2f CE_loss: %.3f LR: %.12f'%(epoch, batch_index+1, len(train_loader), time.time() - t0, np.mean(hist[:,1])*100, np.mean(hist[:,2])*100, np.mean(hist[:,0]), scheduler.get_last_lr()[0]))

    TP_all = np.zeros((2, 1))
    FP_all = np.zeros((2, 1))
    TN_all = np.zeros((2, 1))
    FN_all = np.zeros((2, 1))
    n_valid_sample_all = 0
    F1 = np.zeros((2, 1))
    IoU = np.zeros((2, 1))    

    classifier.eval()
    with torch.no_grad():
        for batch_index, data in enumerate(val_loader):   
        
            # load images, labels and random noise to device (GPU or CPU) 
            images, labels, size, name = data
            images = images[:,[1,2,3,8,10,11],:,:].to(device, dtype=torch.float)  
            labels = labels.unsqueeze(1).to(device)        
            
            # resize images and labels
            images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
            labels_cropped = labels_cropped.squeeze().cpu().numpy()
    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
    
            # extract feature vectors (tokens)
            features = encoder.forward_features(images_cropped.unsqueeze(2)) 
            # print(len(features), features[0].shape)
            
            selected_feats = [features[i] for i in [5, 11, 17, 23]]
            # print(len(selected_feats), selected_feats[0].shape)
                    
            # remove CLS token before passing to segmentation head
            selected_feats = [f[:, 1:, :] for f in selected_feats]
            # print(len(selected_feats), selected_feats[0].shape)
    
            # compute logits
            logits = classifier(selected_feats)
            
            # compute probabilities
            _, predicted = torch.max(logits, 1)
            pred = predicted.squeeze().data.cpu().numpy() 

            TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1), labels_cropped.reshape(-1), 2)
            TP_all += TP
            FP_all += FP
            TN_all += TN
            FN_all += FN
            n_valid_sample_all += n_valid_sample
            
    total_fp = FP_all[1] / n_valid_sample_all # FP when 0: not landslide and 1: landslide
    acc = (TP_all[1] + TN_all[1]) / n_valid_sample_all # Accuracy when 0: not landslide and 1: landslide
    
    OA = np.sum(TP_all)*1.0 / n_valid_sample_all
    for i in range(2):
        P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
        R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
        F1[i] = 2.0*P*R / (P + R + epsilon)
        IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)
    
        logging.info('--' + name_classes[i] + ' Precision: %.2f'%(P * 100))
        logging.info('--' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
        logging.info('--' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
        logging.info('--' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))    

    mF1 = np.mean(F1)   
    mIoU = np.mean(IoU)           
    logging.info('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (landslide): %.2f FP (landslide): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))

    if F1[1]>F1_best:
        F1_best = F1[1]
        # save the model        
        torch.save(classifier.state_dict(), f'{filepath}/model-best.pt')
        logging.info('Best model saved with landslide F1: %.2f'%(F1_best * 100))

    # if mF1>F1_best:
    #     F1_best = mF1
    #     # save the model        
    #     torch.save(classifier.state_dict(), f'{filepath}/model-best.pt')
    #     logging.info('Best model saved with mean F1: %.2f'%(F1_best * 100))
    
    # if mIoU>mIoU_best:
    #     mIoU_best = mIoU
    #     # save the model        
    #     torch.save(classifier.state_dict(), f'{filepath}/model-best.pt')
    #     logging.info('Best model saved with mean IoU: %.2f'%(mIoU_best * 100))
    
# Stop timer
stop = timeit.default_timer()
total_time = stop - start
logging.info(f"Total training time: {int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s")

###############################################################################
# VALIDATION - BEST MODEL
############################################################################### 
saved_state_dict = torch.load(f'{filepath}/model-best.pt')  
classifier.load_state_dict(saved_state_dict)

logging.info("Validation started.")

TP_all = np.zeros((2, 1))
FP_all = np.zeros((2, 1))
TN_all = np.zeros((2, 1))
FN_all = np.zeros((2, 1))
n_valid_sample_all = 0
F1 = np.zeros((2, 1))
IoU = np.zeros((2, 1)) 

classifier.eval()
with torch.no_grad():
    for batch_index, data in enumerate(val_loader):   
        
        # load images, labels and random noise to device (GPU or CPU) 
        images, labels, size, name = data
        images = images[:,[1,2,3,8,10,11],:,:].to(device, dtype=torch.float)  
        labels = labels.unsqueeze(1).to(device)        
        
        # resize images and labels
        images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
        labels_cropped = labels_cropped.squeeze().cpu().numpy()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped.unsqueeze(2)) 
        # print(len(features), features[0].shape)
        
        selected_feats = [features[i] for i in [5, 11, 17, 23]]
        # print(len(selected_feats), selected_feats[0].shape)
                
        # remove CLS token before passing to segmentation head
        selected_feats = [f[:, 1:, :] for f in selected_feats]
        # print(len(selected_feats), selected_feats[0].shape)

        # compute logits
        logits = classifier(selected_feats)
        
        # compute probabilities
        _, predicted = torch.max(logits, 1)
        pred = predicted.squeeze().data.cpu().numpy() 

        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1), labels_cropped.reshape(-1), 2)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

total_fp = FP_all[1] / n_valid_sample_all # FP when 0: not cloud and 1: cloud
acc = (TP_all[1] + TN_all[1]) / n_valid_sample_all # Accuracy when 0: not cloud and 1: cloud

OA = np.sum(TP_all)*1.0 / n_valid_sample_all
for i in range(2):
    P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
    R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
    F1[i] = 2.0*P*R / (P + R + epsilon)
    IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)

    logging.info('--' + name_classes[i] + ' Precision: %.2f'%(P * 100))
    logging.info('--' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
    logging.info('--' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
    logging.info('--' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))    
    
mF1 = np.mean(F1)   
mIoU = np.mean(IoU)           
logging.info('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (landslide): %.2f FP (landslide): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))

###############################################################################
# EVALUATION - BEST MODEL
############################################################################### 
logging.info("Evaluation started.")

TP_all = np.zeros((2, 1))
FP_all = np.zeros((2, 1))
TN_all = np.zeros((2, 1))
FN_all = np.zeros((2, 1))
n_valid_sample_all = 0
F1 = np.zeros((2, 1))
IoU = np.zeros((2, 1)) 

classifier.eval()
with torch.no_grad():
    for batch_index, data in enumerate(test_loader):   
        
        # load images, labels and random noise to device (GPU or CPU) 
        images, labels, size, name = data
        images = images[:,[1,2,3,8,10,11],:,:].to(device, dtype=torch.float)  
        labels = labels.unsqueeze(1).to(device)        
        
        # resize images and labels
        images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
        labels_cropped = labels_cropped.squeeze().cpu().numpy()
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped.unsqueeze(2)) 
        # print(len(features), features[0].shape)
        
        selected_feats = [features[i] for i in [5, 11, 17, 23]]
        # print(len(selected_feats), selected_feats[0].shape)
                
        # remove CLS token before passing to segmentation head
        selected_feats = [f[:, 1:, :] for f in selected_feats]
        # print(len(selected_feats), selected_feats[0].shape)

        # compute logits
        logits = classifier(selected_feats)
        
        # compute probabilities
        _, predicted = torch.max(logits, 1)
        pred = predicted.squeeze().data.cpu().numpy() 

        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1), labels_cropped.reshape(-1), 2)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

total_fp = FP_all[1] / n_valid_sample_all # FP when 0: not landslide and 1: landslide
acc = (TP_all[1] + TN_all[1]) / n_valid_sample_all # Accuracy when 0: not landslide and 1: landslide

OA = np.sum(TP_all)*1.0 / n_valid_sample_all
for i in range(2):
    P = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + epsilon)
    R = TP_all[i]*1.0 / (TP_all[i] + FN_all[i] + epsilon)
    F1[i] = 2.0*P*R / (P + R + epsilon)
    IoU[i] = TP_all[i]*1.0 / (TP_all[i] + FP_all[i] + FN_all[i] + epsilon)

    logging.info('--' + name_classes[i] + ' Precision: %.2f'%(P * 100))
    logging.info('--' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
    logging.info('--' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
    logging.info('--' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))    
    
mF1 = np.mean(F1)   
mIoU = np.mean(IoU)           
logging.info('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (landslide): %.2f FP (landslide): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))

logging.info(f"Training and evaluation complete. Logs saved to training.log in {filepath} directory.")




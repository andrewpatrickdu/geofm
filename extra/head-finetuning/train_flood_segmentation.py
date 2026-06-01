#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that trains a flood detector

python train_flood_segmentation.py --batch_size 32 --num_epochs 300 --encoder /data/Prithvi_EO_V2_300M.pt --encoder_embed_dim 1024 --checkpoint /data/ckpt-flood-segmentation-baseline-1024

python train_flood_segmentation.py --batch_size 32 --num_epochs 300 --encoder /data/ckpt-mae-512/model-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-flood-segmentation-mae-512

python train_flood_segmentation.py --batch_size 32 --num_epochs 300 --encoder /data/ckpt-mae-256/model-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-flood-segmentation-mae-256

python train_flood_segmentation.py --batch_size 32 --num_epochs 300 --encoder /data/ckpt-distillation-512/student-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-flood-segmentation-distillation-512

python train_flood_segmentation.py --batch_size 32 --num_epochs 300 --encoder /data/ckpt-distillation-256/student-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-flood-segmentation-distillation-256

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

name_classes = np.array(['non-water','water'], dtype=str)
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
import rasterio

class Sen1Floods11(Dataset):
    def __init__(self, data, img_path, mask_path, transform = None):
        super().__init__()
        self.data = data.values
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name, mask_name = self.data[index]
        img_path = os.path.join(self.img_path, img_name)
        mask_path = os.path.join(self.mask_path, mask_name)
        
        with rasterio.open(img_path) as src:
            image = src.read()  # shape: (bands, height, width)
        image = image.astype(np.float32) / 10000.
        
        with rasterio.open(mask_path) as src:
            mask = src.read()  # shape: (bands, height, width)
        
        # if self.transform is not None:
            # image = self.transform(image)
            # mask = self.transform(mask)[0]
        
        return image, mask, img_name, mask_name

img_dir = '/data/flood-detection/v1.1/data/flood_events/HandLabeled/S2Hand'
mask_dir = '/data/flood-detection/v1.1/data/flood_events/HandLabeled/LabelHand'

training_dir = '/data/flood-detection/v1.1/splits/flood_handlabeled/flood_train_data.csv'
validation_dir = '/data/flood-detection/v1.1/splits/flood_handlabeled/flood_valid_data.csv'
testing_dir = '/data/flood-detection/v1.1/splits/flood_handlabeled/flood_test_data.csv'
    
# dataset splits
training = pd.read_csv(training_dir, header=None)
training = training.applymap(lambda x: x.replace("S1", "S2"))

validation = pd.read_csv(validation_dir, header=None)
validation = validation.applymap(lambda x: x.replace("S1", "S2"))

testing = pd.read_csv(testing_dir, header=None)
testing = testing.applymap(lambda x: x.replace("S1", "S2"))

# Image transformations
train_transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                     # AddGaussianNoise(0., 1.),
                                     # transforms.ToPILImage(),
                                     # transforms.RandomHorizontalFlip(p=0.5), 
                                     # transforms.RandomVerticalFlip(p=0.5),   
                                     # transforms.ToTensor(),
                                     # transforms.RandomAdjustSharpness(sharpness_factor, p=0.5),
                                     # transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0)),
                                     # transforms.Resize((356, 356)),
                                     # transforms.RandomCrop((299, 299)),
                                     ])

valid_transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                     # transforms.ToPILImage(),
                                     # transforms.RandomHorizontalFlip(p=0.5),
                                     # transforms.ToTensor(),
                                     ])

test_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                    ])

batch_size = args.batch_size
train_dataset = Sen1Floods11(training, img_dir, mask_dir, train_transform)
val_dataset = Sen1Floods11(validation, img_dir, mask_dir, valid_transform)
test_dataset = Sen1Floods11(testing, img_dir, mask_dir, test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
logging.info(f"Loaded training dataset with {len(train_dataset)} samples.")
logging.info(f"Loaded validation dataset with {len(val_dataset)} samples.")
logging.info(f"Loaded testing dataset with {len(test_dataset)} samples.")

# # CHECKER: total number of water and not water pixels in dataset
# water_pixels = 0
# not_water_pixels = 0
# missing_data = 0
# for _, mask, _, _ in test_loader:

#     mask = mask.numpy()

#     water_pixels += np.sum(mask == 1)
#     not_water_pixels += np.sum(mask == 0)
#     missing_data += np.sum(mask == -1)
# print(f"- Total water pixels: {water_pixels}")
# print(f"- Total not water pixels: {not_water_pixels}")
# print(f"- Total missing value pixels: {missing_data}")
# print(f"- Total pixels: {water_pixels + not_water_pixels + missing_data}")


'''
Training: 
- Total water pixels: 5,447,035 (8%)
- Total not water pixels: 51,856,372 (78%)
- Total missing value pixels: 8,756,881 (14%)
- Total pixels: 66,060,288


Validation
- Total water pixels: 2,237,605 (10%)
- Total not water pixels: 18,057,120 (77%)
- Total missing value pixels: 3,036,091 (13%)
- Total pixels: 23,330,816


Testing
- Total water pixels: 2,566,101 (11%)
- Total not water pixels: 17,951,266 (76%)
- Total missing value pixels: 3,075,593 (13%)
- Total pixels: 23,592,960
'''

###############################################################################
# ENCODER MODEL
###############################################################################
# Initialize model
encoder = PrithviViT(img_size=224, in_chans = 4, embed_dim = args.encoder_embed_dim).to(device)
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
import torch
import torch.nn as nn

# UperNet 
class PPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        pool_scales = [14, 7, 5, 2]  # input 14x14 → outputs of 1x1, 2x2, 3x3, 6x6

        for scale in pool_scales:
            # Downsampling stage
            self.stages.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=scale, stride=scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

            # Upsampling stage: each must map back to 14x14
            upscale = 14 // (14 // scale)  # e.g. 1x1 → 14x14 ⇒ 14x
            self.upsamples.append(nn.ConvTranspose2d(
                out_channels, out_channels,
                kernel_size=upscale, stride=upscale
            ))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + 4 * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        ppm_outs = [x]
        for i, stage in enumerate(self.stages):
            out = stage(x)
            out = self.upsamples[i](out)
            ppm_outs.append(out)
        return self.bottleneck(torch.cat(ppm_outs, dim=1))


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=8, stride=8)

    def forward(self, feats):
        fpn_feats = []
        for i in range(len(feats)):
            x = self.lateral_convs[i](feats[i])
            x = self.fpn_convs[i](x)
            x = self.upsample(x)  # 14x14 → 112x112
            fpn_feats.append(x)
        return torch.cat(fpn_feats, dim=1)  # [B, C*4, 112, 112]


class UPerNetDecoder(nn.Module):
    def __init__(self, input_dim=256, num_classes=2, ppm_out_channels=80, fpn_out_channels=64):
        super().__init__()
        self.num_levels = 4
        self.ppm = PPM(input_dim, ppm_out_channels)
        self.fpn = FPN(
            in_channels_list=[ppm_out_channels] + [input_dim] * (self.num_levels - 1),
            out_channels=fpn_out_channels
        )
        self.classifier = nn.Sequential(
            nn.ConvTranspose2d(fpn_out_channels * self.num_levels, num_classes, kernel_size=2, stride=2)  # 112 → 224
        )

    def forward(self, x):  # x: [B, 4, 196, 256]
        B, L, N, D = x.shape
        feats = []
        H = W = 14  # sqrt(196)

        for i in range(L):
            tokens = x[:, i]  # [B, 196, D]
            f = tokens.permute(0, 2, 1).reshape(B, D, H, W)  # [B, D, 14, 14]
            feats.append(f)

        feats[0] = self.ppm(feats[0])      # Apply PPM to top-level feature
        fpn_out = self.fpn(feats)          # FPN fusion and upsampling
        out = self.classifier(fpn_out)     # Final 112 → 224
        return out  # [B, num_classes, 224, 224]

classifier = UPerNetDecoder(input_dim=encoder.embed_dim, num_classes=2).to(device)

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
    weight=weight, size_average=None, ignore_index=-1, reduce=None, reduction='mean')

# Define optimiser
optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-5, weight_decay=5e-4)

# Define learning rate
num_epochs = args.num_epochs
num_warmup_epochs = num_epochs*0.1 # 10% of total epochs

lr_start = 1e-5
lr_max = 5e-4

from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(epoch):
    if epoch < num_warmup_epochs:
        # Linear warm-up
        return (lr_start + (lr_max - lr_start) * epoch / num_warmup_epochs) / lr_max
    else:
        # Cosine annealing schedule after warm-up
        progress = (epoch - num_warmup_epochs) / (num_epochs - num_warmup_epochs)
        return 0.5 * (1. + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


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
        images, labels, img_name, mask_name = data
        images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float) 
        # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float) 
        labels = labels.to(device)      
        
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
        # VIZ_FACTOR = 2.5
        # idx = 10
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

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped.unsqueeze(2)) # torch.Size([20, 24, 197, 256])
        
        # extract features from 6th, 12th, 18th, and last layers and remove CLS token before passing to segmentation head
        features = features[:,[5, 11, 17, 23],:,:] # torch.Size([20, 4, 197, 256])
        features = features[:,:,1:,:] # torch.Size([20, 4, 196, 256])
        
        # compute logits
        logits = classifier(features)
        
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
            images, labels, img_name, mask_name = data
            images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float) 
            # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float)  
            labels = labels.to(device)     
            
            # resize images and labels
            images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
            labels_cropped = labels_cropped.squeeze().cpu().numpy()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
    
            # extract feature vectors (tokens)
            features = encoder.forward_features(images_cropped.unsqueeze(2)) 
            
            # extract features from 6th, 12th, 18th, and last layers and remove CLS token before passing to segmentation head
            features = features[:,[5, 11, 17, 23],:,:] # torch.Size([20, 4, 197, 256])
            features = features[:,:,1:,:] # torch.Size([20, 4, 196, 256])
    
            # compute logits
            logits = classifier(features)
            
            # compute probabilities
            _, predicted = torch.max(logits, 1)
            pred = predicted.squeeze().data.cpu().numpy() 

            TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1), labels_cropped.reshape(-1), 2)
            TP_all += TP
            FP_all += FP
            TN_all += TN
            FN_all += FN
            n_valid_sample_all += n_valid_sample
            
    total_fp = FP_all[1] / n_valid_sample_all # FP when 0: not water and 1: water
    acc = (TP_all[1] + TN_all[1]) / n_valid_sample_all # Accuracy when 0: not water and 1: water
    
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
    logging.info('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (water): %.2f FP (water): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))
    
    if mIoU>mIoU_best:
        mIoU_best = mIoU
        # save the model        
        model_name = 'best_model.pt'
        torch.save(classifier.state_dict(), f'{filepath}/model-best.pt')
        logging.info('Best model saved with mean IoU: %.2f'%(mIoU_best * 100))
    
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
        images, labels, img_name, mask_name = data
        images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float)
        # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float)  
        labels = labels.to(device)     
        
        # resize images and labels
        images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
        labels_cropped = labels_cropped.squeeze().cpu().numpy()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped.unsqueeze(2)) 
    
        # extract features from 6th, 12th, 18th, and last layers and remove CLS token before passing to segmentation head
        features = features[:,[5, 11, 17, 23],:,:] # torch.Size([20, 4, 197, 256])
        features = features[:,:,1:,:] # torch.Size([20, 4, 196, 256])

        # compute logits
        logits = classifier(features)
        
        # compute probabilities
        _, predicted = torch.max(logits, 1)
        pred = predicted.squeeze().data.cpu().numpy() 

        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1), labels_cropped.reshape(-1), 2)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

total_fp = FP_all[1] / n_valid_sample_all # FP when 0: not water and 1: water
acc = (TP_all[1] + TN_all[1]) / n_valid_sample_all # Accuracy when 0: not water and 1: water

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
logging.info('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (water): %.2f FP (water): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))

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
        images, labels, img_name, mask_name = data
        images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float) 
        # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float)  
        labels = labels.to(device)    
        
        # resize images and labels
        images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
        labels_cropped = labels_cropped.squeeze().cpu().numpy()
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped.unsqueeze(2)) 

        # extract features from 6th, 12th, 18th, and last layers and remove CLS token before passing to segmentation head
        features = features[:,[5, 11, 17, 23],:,:] # torch.Size([20, 4, 197, 256])
        features = features[:,:,1:,:] # torch.Size([20, 4, 196, 256])

        # compute logits
        logits = classifier(features)
        
        # compute probabilities
        _, predicted = torch.max(logits, 1)
        pred = predicted.squeeze().data.cpu().numpy() 

        TP,FP,TN,FN,n_valid_sample = eval_image(pred.reshape(-1), labels_cropped.reshape(-1), 2)
        TP_all += TP
        FP_all += FP
        TN_all += TN
        FN_all += FN
        n_valid_sample_all += n_valid_sample

total_fp = FP_all[1] / n_valid_sample_all # FP when 0: not water and 1: water
acc = (TP_all[1] + TN_all[1]) / n_valid_sample_all # Accuracy when 0: not water and 1: water

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
logging.info('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (water): %.2f FP (water): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))

logging.info(f"Training and evaluation complete. Logs saved to training.log in {filepath} directory.")




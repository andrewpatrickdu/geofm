#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that trains a cloud detector

python train_cloud_segmentation.py --batch_size 32 --num_epochs 120 --encoder /data/Prithvi_EO_V2_300M.pt --encoder_embed_dim 1024 --checkpoint /data/ckpt-cloud-segmentation-baseline-1024

python train_cloud_segmentation.py --batch_size 32 --num_epochs 120 --encoder /data/ckpt-mae-512/model-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-cloud-segmentation-mae-512

python train_cloud_segmentation.py --batch_size 32 --num_epochs 120 --encoder /data/ckpt-mae-256/model-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-cloud-segmentation-mae-256

python train_cloud_segmentation.py --batch_size 32 --num_epochs 120 --encoder /data/ckpt-distillation-512/student-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-cloud-segmentation-distillation-512

python train_cloud_segmentation.py --batch_size 32 --num_epochs 120 --encoder /data/ckpt-distillation-256/student-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-cloud-segmentation-distillation-256

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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from prithvi_mae import PrithviViT

name_classes = np.array(['non-cloud','cloud'], dtype=str)
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
class S2CloudMaskCatalogue(Dataset):
    def __init__(self, data, img_path, mask_path, transform = None):
        super().__init__()
        self.data = data.values
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.img_path, img_name)
        mask_path = os.path.join(self.mask_path, img_name)
        
        image = np.load(img_path)
        mask = np.load(mask_path)
        
        if self.transform is not None:
            image = self.transform(image)
            # mask = self.transform(mask)[0]
        
        return image, mask, label

img_dir = '/data/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/images'
mask_dir = '/data/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/cloudmasks'
lab_dir_70 = '/data/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/labels/TF70.csv'
    
# TF70 dataset
labels_70 = pd.read_csv(lab_dir_70)
labels_70 = labels_70.sample(frac=1, random_state=0)
N = min(labels_70['is_cloudy'].value_counts()[0],
        labels_70['is_cloudy'].value_counts()[1])
cloudy = labels_70.loc[labels_70['is_cloudy'] == 1]
not_cloudy = labels_70.loc[labels_70['is_cloudy'] == 0]

training_70 = pd.concat([cloudy[0:int(0.70*N)], not_cloudy[0:int(0.70*N)]])
validation_70 = pd.concat([cloudy[int(0.70*N):int(0.85*N)], not_cloudy[int(0.70*N):int(0.85*N)]])
test_70 = pd.concat([cloudy[int(0.85*N):int(len(cloudy)*1.00)], not_cloudy[int(0.85*N):int(len(not_cloudy)*1.00)]])

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
train_dataset = S2CloudMaskCatalogue(training_70, img_dir, mask_dir, train_transform)
val_dataset = S2CloudMaskCatalogue(validation_70, img_dir, mask_dir, valid_transform)
test_dataset = S2CloudMaskCatalogue(test_70, img_dir, mask_dir, test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
logging.info(f"Loaded training dataset with {len(train_dataset)} samples.")
logging.info(f"Loaded validation dataset with {len(val_dataset)} samples.")
logging.info(f"Loaded testing dataset with {len(test_dataset)} samples.")

# # CHECKER: total number of cloudy and not cloudy pixels in dataset
# cloudy_pixels = 0
# not_cloudy_pixels = 0
# for _, mask, _ in test_loader:

#     mask = mask.numpy()

#     cloudy_pixels += np.sum(mask == 255)
#     not_cloudy_pixels += np.sum(mask == 0)

# print(f"Total cloudy pixels: {cloudy_pixels}")
# print(f"Total not cloudy pixels: {not_cloudy_pixels}")
# print(f"Total pixels: {cloudy_pixels + not_cloudy_pixels}")

'''
Training: 
- Total cloudy pixels: 195,661,795 (55%)
- Total not cloudy pixels: 159,805,469 (45%)
- Total pixels: 355,467,264

Validation
- Total cloudy pixels: 41,193,973 (54%)
- Total not cloudy pixels: 34,827,787 (46%)
- Total pixels: 76,021,760

Testing
- Total cloudy pixels: 45,978,242
- Total not cloudy pixels: 60,452,222
- Total pixels: 106,430,464

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
import torch.nn as nn

# UNet (with skip connections)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        self.reduce_channels = 80

        # Project each of the 4 ViT feature levels into spatial feature maps
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(input_dim, self.reduce_channels, kernel_size=1) for _ in range(4)
        ])

        # Main upsample path
        self.up1 = Up(self.reduce_channels, self.reduce_channels, self.reduce_channels // 2)  # 14 → 28
        self.up2 = Up(self.reduce_channels // 2, self.reduce_channels, self.reduce_channels // 4)  # 28 → 56
        self.up3 = Up(self.reduce_channels // 4, self.reduce_channels, self.reduce_channels // 8)  # 56 → 112
        self.up4 = nn.Sequential(  # 112 → 224
            nn.ConvTranspose2d(self.reduce_channels // 8, self.reduce_channels // 8, kernel_size=2, stride=2),
            nn.Conv2d(self.reduce_channels // 8, num_classes, kernel_size=1)
        )

        # Static skip upsampling layers to match spatial sizes
        self.skip_upsample_2 = nn.ConvTranspose2d(self.reduce_channels, self.reduce_channels, kernel_size=2, stride=2)   # 14 → 28
        self.skip_upsample_1 = nn.ConvTranspose2d(self.reduce_channels, self.reduce_channels, kernel_size=4, stride=4)   # 14 → 56
        self.skip_upsample_0 = nn.ConvTranspose2d(self.reduce_channels, self.reduce_channels, kernel_size=8, stride=8)   # 14 → 112

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):  # x: [B, 4, 196, 256]
        B, num_levels, N, D = x.shape
        H = W = 14

        # Process and reduce each level from ViT
        features = []
        for i in range(num_levels):
            feat = x[:, i]  # [B, 196, D]
            feat = feat.permute(0, 2, 1).reshape(B, D, H, W)  # [B, D, 14, 14]
            features.append(self.lateral_convs[i](feat))     # [B, 80, 14, 14]

        # Main upsample path with skip connections
        x = features[-1]  # features[3]
        x = self.up1(x, self.skip_upsample_2(features[2]))  # 14 → 28
        x = self.dropout(x)

        x = self.up2(x, self.skip_upsample_1(features[1]))  # 28 → 56
        x = self.dropout(x)

        x = self.up3(x, self.skip_upsample_0(features[0]))  # 56 → 112
        x = self.dropout(x)

        x = self.up4(x)  # 112 → 224
        return x  # [B, num_classes, 224, 224]


classifier = UNetDecoder(input_dim=encoder.embed_dim, num_classes=2).to(device)


logging.info("Decoder initialized.")

# CHCKER: Display total number of parameters
total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
logging.info(f"Decoder has {total_params:,} parameters.")

###############################################################################
# TRAINING PARAMETERS
###############################################################################  
# Define loss
weight = torch.tensor([2., 1.]).cuda()
criterion = nn.CrossEntropyLoss(
    weight=weight, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

# Define optimiser
optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-5, weight_decay=5e-4)

# Define learning rate
num_epochs = args.num_epochs
num_warmup_epochs = num_epochs*0.1 # 10% of total epochs

lr_start = 1e-5
lr_max = 5e-4

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
for epoch in range(1, num_epochs + 1):
    
    # start timer
    t0 = time.time()

    hist = np.zeros((len(train_loader), 3))
    
    encoder.eval()
    classifier.train()
    for batch_index, data in enumerate(train_loader):   
 
        # load images, labels and random noise to device (GPU or CPU) 
        images, labels, labels_70 = data  
        images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float)  
        # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float) 
        labels = (labels == 255).unsqueeze(1).to(device)       
        random_noise = torch.rand(images.size(0), encoder.sequence_length-1).to(device)

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
        # idx = 1
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
        features = encoder.forward_features(images_cropped.unsqueeze(2)) # torch.Size([32, 24, 197, 256])

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
    scheduler.step()

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
            images, labels, labels_70 = data
            # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float) 
            images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float)  
            labels = (labels == 255).unsqueeze(1).to(device)     
            random_noise = torch.rand(images.size(0), encoder.sequence_length-1).to(device)
            
            # resize images and labels
            images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
            labels_cropped = labels_cropped.squeeze().cpu().numpy()
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
    
            # extract feature vectors (tokens)
            features = encoder.forward_features(images_cropped.unsqueeze(2)) # print(len(features), features[0].shape)
            
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
    logging.info('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (cloud): %.2f FP (cloud): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))
    
    if mF1>F1_best:
        F1_best = mF1
        # save the model        
        model_name = 'best_model.pt'
        torch.save(classifier.state_dict(), f'{filepath}/model-best.pt')
        logging.info('Best model saved with mean F1: %.2f'%(F1_best * 100))
    
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
        images, labels, labels_70 = data
        # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float)  
        images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float)  
        labels = (labels == 255).unsqueeze(1).to(device)     
        random_noise = torch.rand(images.size(0), encoder.sequence_length-1).to(device)
        
        # resize images and labels
        images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
        labels_cropped = labels_cropped.squeeze().cpu().numpy()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped.unsqueeze(2)) # print(len(features), features[0].shape)

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
logging.info('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (cloud): %.2f FP (cloud): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))

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
        images, labels, labels_70 = data
        # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float)   
        images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float)  
        labels = (labels == 255).unsqueeze(1).to(device)     
        random_noise = torch.rand(images.size(0), encoder.sequence_length-1).to(device)
        
        # resize images and labels
        images_cropped = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        labels_cropped = F.interpolate(labels.float(), size=(224, 224), mode='nearest').long()
        labels_cropped = labels_cropped.squeeze().cpu().numpy()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped.unsqueeze(2)) # print(len(features), features[0].shape)
        
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
logging.info('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (cloud): %.2f FP (cloud): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))

logging.info(f"Training and evaluation complete. Logs saved to training.log in {filepath} directory.")




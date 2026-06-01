#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that trains a cloud detector.

# Prithvi v2 (with embedding dimension of 256) pretrained via dual-MAE distillation
python evaluate_cloud_segmentation.py --encoder /home/andrew/github/GFM/ckpt-distillation-256/student-final.pt --encoder_embed_dim 256 --classifier /home/andrew/github/GFM/ckpt-cloud-segmentation-distillation-256/model-best.pt

### EXPECTED OUTCOME ###
--- mIoU: 83.84 mean F1: 91.21 OA: 91.21 ACC (cloud): 91.21 FP (cloud): 1.63

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

# set random seeds for reproducibility
seed = 0
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
torch.cuda.manual_seed(seed) 
np.random.seed(seed) 
random.seed(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parse command-line arguments
import argparse
parser = argparse.ArgumentParser(description='DUA')
parser.add_argument('--encoder', type=str, default='/home/andrew/GFM/ckpt-mae-256/model-final.pt', help='Directory of encoder pretrained weights')
parser.add_argument('--encoder_embed_dim', type=int, default=256, help='Encoder embedding dimension')
parser.add_argument('--classifier', type=str, default='/home/andrew/GFM/ckpt-cloud-classification-mae-256/model-final.pt', help='Directory of classifier pretrained weights')
args = parser.parse_args()

# Log the start of the script
print("Script started.")

# Set the device to GPU ("cuda") if available; otherwise, default to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

img_dir = '/home/andrew/cloud-detector/datasets/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/images'
mask_dir = '/home/andrew/cloud-detector/datasets/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/cloudmasks'
lab_dir_70 = '/home/andrew/cloud-detector/datasets/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/labels/TF70.csv'
    
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

batch_size = 1
train_dataset = S2CloudMaskCatalogue(training_70, img_dir, mask_dir, train_transform)
val_dataset = S2CloudMaskCatalogue(validation_70, img_dir, mask_dir, valid_transform)
test_dataset = S2CloudMaskCatalogue(test_70, img_dir, mask_dir, test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
print(f"Loaded training dataset with {len(train_dataset)} samples.")
print(f"Loaded validation dataset with {len(val_dataset)} samples.")
print(f"Loaded testing dataset with {len(test_dataset)} samples.")

###############################################################################
# ENCODER MODEL
###############################################################################
# Initialize model
encoder = PrithviViT(img_size=224, in_chans = 4, embed_dim = args.encoder_embed_dim).to(device)
print("Encoder initialized.")

# CHCKER: Display total number of parameters
total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"Encoder has {total_params:,} parameters.")

# Load checkpoint (pretrained weights)
checkpoint = args.encoder
state_dict = torch.load(checkpoint, map_location=device)

# Discard fixed pos_embedding weight
for k in list(state_dict.keys()):
    if 'pos_embed' in k:
        del state_dict[k]
        
encoder.load_state_dict(state_dict, strict=False)
print(f"Loaded checkpoint from {checkpoint}.")

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
print(f"Decoder has {total_params:,} parameters.")

# Load checkpoint (pretrained weights)
checkpoint = args.classifier
state_dict = torch.load(checkpoint, map_location=device)

classifier.load_state_dict(state_dict)
print(f"Loaded checkpoint from {checkpoint}.")

###############################################################################
# EVALUATION
############################################################################### 
name_classes = np.array(['non-cloud','cloud'], dtype=str)
epsilon = 1e-14

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


print("Evaluation started.")

TP_all = np.zeros((2, 1))
FP_all = np.zeros((2, 1))
TN_all = np.zeros((2, 1))
FN_all = np.zeros((2, 1))
n_valid_sample_all = 0
F1 = np.zeros((2, 1))
IoU = np.zeros((2, 1)) 

i = 1

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

    print('--' + name_classes[i] + ' Precision: %.2f'%(P * 100))
    print('--' + name_classes[i] + ' Recall: %.2f'%(R * 100))            
    print('--' + name_classes[i] + ' IoU: %.2f'%(IoU[i] * 100))              
    print('--' + name_classes[i] + ' F1: %.2f'%(F1[i] * 100))    
    
mF1 = np.mean(F1)   
mIoU = np.mean(IoU)           
print('--- mIoU: %.2f mean F1: %.2f OA: %.2f ACC (cloud): %.2f FP (cloud): %.2f'%(mIoU * 100, mF1 * 100, OA * 100, acc * 100, total_fp * 100))



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that trains an agb estimator

python train_agb_estimation.py --batch_size 32 --num_epochs 200 --encoder /data/Prithvi_EO_V2_300M.pt --encoder_embed_dim 1024 --checkpoint /data/ckpt-agb-estimation-baseline-1024

python train_agb_estimation.py --batch_size 32 --num_epochs 200 --encoder /data/ckpt-mae-512/model-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-agb-estimation-mae-512

python train_agb_estimation.py --batch_size 32 --num_epochs 200 --encoder /data/ckpt-mae-256/model-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-agb-estimation-mae-256

python train_agb_estimation.py --batch_size 32 --num_epochs 200 --encoder /data/ckpt-distillation-512/student-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-agb-estimation-distillation-512

python train_agb_estimation.py --batch_size 32 --num_epochs 200 --encoder /data/ckpt-distillation-256/student-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-agb-estimation-distillation-256

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
from sklearn.model_selection import train_test_split

class BioMassters(Dataset):
    def __init__(self, dataframe, img_path, mask_path, transform=None, crop_size=224, num_timesteps=12):
        super().__init__()
        self.transform = transform
        self.img_path = img_path
        self.mask_path = mask_path
        self.crop_size = crop_size
        self.num_timesteps = num_timesteps

        # Group all files by chip_id and corresponding_agbm (columns 1 and 10)
        self.grouped = dataframe.groupby([dataframe.columns[1], dataframe.columns[10]])[dataframe.columns[0]].apply(list).reset_index()

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, index):
        chip_id, agb_name, filenames = self.grouped.iloc[index]

        # Sort filenames to ensure consistent time order
        filenames = sorted(filenames)

        images = []
        for filename in filenames:
            img_path = os.path.join(self.img_path, filename)
            with rasterio.open(img_path) as src:
                img = src.read().astype(np.float32)
            # Min-max normalize each image separately
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            images.append(img)

        # Pad with zeros if less than required timesteps
        if len(images) < self.num_timesteps:
            pad_count = self.num_timesteps - len(images)
            pad_shape = images[0].shape
            padding = [np.zeros(pad_shape, dtype=np.float32) for _ in range(pad_count)]
            images += padding
        elif len(images) > self.num_timesteps:
            images = images[:self.num_timesteps]

        # Stack images along a new time dimension (T, Bands, H, W)
        image_stack = np.stack(images, axis=0)
        # Rearrange to shape (Bands, T, H, W)
        image_stack = np.transpose(image_stack, (1, 0, 2, 3))

        # Read AGB mask
        agb_path = os.path.join(self.mask_path, agb_name)
        with rasterio.open(agb_path) as src:
            agb = src.read().astype(np.float32)

        # Optional cloud mask (example: last band from each image)
        cloud_mask = np.stack([img[-1] for img in images], axis=0)  # shape: (T, H, W)

        return image_stack, agb, cloud_mask, chip_id

train_img_dir = '/data/agb-estimation/train_features'
train_mask_dir = '/data/agb-estimation/train_agbm'
test_img_dir = '/data/agb-estimation/test_features'
test_mask_dir = '/data/agb-estimation/test_agbm'

training_dir = '/data/agb-estimation/The_BioMassters_-_features_metadata.csv.csv'
testing_dir = '/data/agb-estimation/The_BioMassters_-_features_metadata.csv.csv'

# TRAINING AND VALIDATION SETS    
# Filter out S1 and test images
df = pd.read_csv(training_dir, header=None)
df = df[~df[0].str.contains("S1")] # remove rows where 'filename' contains 'S1'
df = df[~df[7].str.contains("test")] # remove rows where 's3path' contains 'test'

# Perform 80/20 split on unique chip IDs
unique_chip_ids = df[1].unique() # total: 8690
train_ids, val_ids = train_test_split(unique_chip_ids, test_size=0.2, random_state=42) # train/val: 6952/1738

# Split full dataframe using chip_id
training = df[df[1].isin(train_ids)].reset_index(drop=True)
validation = df[df[1].isin(val_ids)].reset_index(drop=True)
validation = validation.iloc[1:].reset_index(drop=True)

# TESTING SET
# Filter out S1 and train images
df = pd.read_csv(testing_dir, header=None)
df = df[~df[0].str.contains("S1")] # remove rows where 'filename' contains 'S1'
df = df[~df[7].str.contains("train")] # remove rows where 's3path' contains 'train'
test_ids = df[1].unique() # total: 2774

testing = df[df[1].isin(test_ids)].reset_index(drop=True)
testing = testing.iloc[1:].reset_index(drop=True)

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
train_dataset = BioMassters(training, train_img_dir, train_mask_dir, train_transform)
val_dataset = BioMassters(validation, train_img_dir, train_mask_dir, valid_transform)
test_dataset = BioMassters(testing, test_img_dir, test_mask_dir, test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
logging.info(f"Loaded training dataset with {len(train_dataset)} samples.")
logging.info(f"Loaded validation dataset with {len(val_dataset)} samples.")
logging.info(f"Loaded testing dataset with {len(test_dataset)} samples.")

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

# UperNet - Regression - 11.5MB
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
    def __init__(self, input_dim=256, num_outputs=1, ppm_out_channels=80, fpn_out_channels=64):
        super().__init__()
        self.num_levels = 4
        self.ppm = PPM(input_dim, ppm_out_channels)
        self.fpn = FPN(
            in_channels_list=[ppm_out_channels] + [input_dim] * (self.num_levels - 1),
            out_channels=fpn_out_channels
        )
        self.regressor = nn.ConvTranspose2d(
            fpn_out_channels * self.num_levels,
            num_outputs,
            kernel_size=2,
            stride=2  # 112 → 224
        )

    def forward(self, x):  # x: [B, 4, 196*T, D]
        B, L, NT, D = x.shape
        T = NT // 196
        H = W = 14
    
        feats = []
    
        for i in range(L):
            tokens = x[:, i]  # [B, 196*T, D]
            tokens = tokens.view(B, T, 196, D)  # [B, T, 196, D]
            tokens = tokens.mean(dim=1)  # temporal average → [B, 196, D]
            f = tokens.permute(0, 2, 1).reshape(B, D, H, W)  # [B, D, 14, 14]
            feats.append(f)
    
        feats[0] = self.ppm(feats[0])
        fpn_out = self.fpn(feats)
        out = self.regressor(fpn_out)
        return out  # [B, 1, 224, 224]

classifier = UPerNetDecoder(input_dim=encoder.embed_dim, num_outputs=1).to(device)

###############################################################################
logging.info("Decoder initialized.")

# CHCKER: Display total number of parameters
total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
logging.info(f"Decoder has {total_params:,} parameters.")

###############################################################################
# TRAINING PARAMETERS
###############################################################################  
def rmse_per_image(preds, targets):
    # preds/targets shape: [B, 1, H, W]
    mask = targets != 0  # mask out zero pixels
    squared_error = (preds - targets) ** 2

    # Avoid division by zero by setting denominator to at least 1
    masked_sum = torch.sum(squared_error * mask, dim=(1, 2, 3))
    valid_pixels = torch.sum(mask, dim=(1, 2, 3)).clamp(min=1)

    batch_rmse = torch.sqrt(masked_sum / valid_pixels)
    return batch_rmse

# Define optimiser
optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-4, weight_decay=0.05)

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
def random_crop(imgs, labs, crop_size=224):
    """
    Randomly crop a 5D image tensor [B, C, T, H, W] and a 4D label tensor [B, 1, H, W]
    to [B, C, T, crop_size, crop_size] and [B, 1, crop_size, crop_size] respectively.
    """
    B, C, T, H, W = imgs.shape
    _, _, H_l, W_l = labs.shape

    assert H == H_l and W == W_l, "Image and label spatial dimensions must match"

    top = random.randint(0, H - crop_size)
    left = random.randint(0, W - crop_size)

    cropped_images = imgs[:, :, :, top:top+crop_size, left:left+crop_size]
    cropped_labels = labs[:, :, top:top+crop_size, left:left+crop_size]

    return cropped_images, cropped_labels

def center_crop(imgs, labs, crop_size):
    # images: [B, C, T, H, W], labels: [B, 1, H, W]
    _, _, _, H, W = imgs.shape
    top = (H - crop_size) // 2
    left = (W - crop_size) // 2
    images_cropped = imgs[:, :, :, top:top+crop_size, left:left+crop_size]
    labels_cropped = labs[:, :, top:top+crop_size, left:left+crop_size]
    return images_cropped, labels_cropped

# Start timer
start = timeit.default_timer()
logging.info("Training started.")
logging.info(f"Training Parameters - Batch Size: {batch_size}, Epochs: {num_epochs}, Encoder embed_dim: {encoder.embed_dim}")
 
# Perform fine-tuning
best_val_rmse = float("inf")
for epoch in range(1, num_epochs + 1):
    
    # start timer
    t0 = time.time()

    train_rmses = []
    val_rmses = []
    
    encoder.eval()
    classifier.train()
    for batch_index, data in enumerate(train_loader): 
 
        # load images, labels and random noise to device (GPU or CPU) 
        images, labels, cloud_mask, chip_id = data
        
        # images = images[:,[0,1,2,7,8,9],:,:,:].to(device, dtype=torch.float)
        images = images[:,[0,1,2,7],:,:,:].to(device, dtype=torch.float)
        labels = labels.to(device)      

        # crop images and labels
        images_cropped, labels_cropped = random_crop(images, labels, 224)
        # labels_cropped = labels_cropped.squeeze(1)
        
        images_cropped = images_cropped.clone()
        labels_cropped = labels_cropped.clone()

        # Apply same random flipping
        batch_size = images_cropped.shape[0]
        for i in range(batch_size):
            if torch.rand(1) < 0.5:  # Horizontal flip
                images_cropped[i] = torch.flip(images_cropped[i], dims=[-1])  # Flip W
                labels_cropped[i] = torch.flip(labels_cropped[i], dims=[-1])
        
            if torch.rand(1) < 0.5:  # Vertical flip
                images_cropped[i] = torch.flip(images_cropped[i], dims=[-2])  # Flip H
                labels_cropped[i] = torch.flip(labels_cropped[i], dims=[-2])


        # # CHECKER - plot image
        # VIZ_FACTOR = 2.5
        # idx = 1
        # month = 0 
        # img = images[idx,:,month,:,:].permute(1, 2, 0).cpu().numpy() * VIZ_FACTOR
        # mask = labels[idx].cpu().numpy()
        # mask = mask.squeeze(0)
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(img[:, :, [2, 1, 0]])  # Assuming BGR, swap to RGB
        # axes[0].set_title("Image")
        # # axes[0].axis("off")
        # heatmap = axes[1].imshow(mask, cmap='viridis', interpolation='nearest')  # Ensure correct scaling
        # axes[1].set_title("Above-Ground Biomass (AGB) Heatmap")
        # # axes[1].axis("off")
        # plt.colorbar(heatmap, label='AGB (t/ha)')
        # plt.tight_layout()
        # plt.show()

        # # CHECKER - plot cropped image
        # img = images_cropped[idx,:,month,:,:].permute(1, 2, 0).cpu().numpy() * VIZ_FACTOR
        # mask = labels_cropped[idx].cpu().numpy()
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(img[:, :, [2, 1, 0]])  # Assuming BGR, swap to RGB
        # axes[0].set_title("Image")
        # # axes[0].axis("off")
        # heatmap = axes[1].imshow(mask.squeeze(0), cmap='viridis', interpolation='nearest')  # Ensure correct scaling
        # axes[1].set_title("Above-Ground Biomass (AGB) Heatmap")
        # # axes[1].axis("off")
        # plt.colorbar(heatmap, label='AGB (t/ha)')
        # plt.tight_layout()
        # plt.show()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped) # 24, torch.Size([1, 1765, 256])
        # print(len(features), features[0].shape)
        
        # extract features from 6th, 12th, 18th, and last layers and remove CLS token before passing to segmentation head
        features = features[:,[5, 11, 17, 23],:,:] # torch.Size([20, 4, 197, 256])
        features = features[:,:,1:,:] # torch.Size([20, 4, 196, 256])

        # compute predictions
        pred = classifier(features)
        
        # compute per-image RMSEs
        batch_rmses = rmse_per_image(pred, labels_cropped)  # shape: [B]
        train_rmses.extend(batch_rmses.tolist())
        
        # compute loss 
        loss = batch_rmses.mean()

        # backward propagation: compute gradient of the loss wrt model parameters
        loss.backward()

        # update the model parameters
        optimizer.step()
        
        # update learning rate
        scheduler.step(epoch + batch_index / len(train_loader)) 
        
        #logging.info(batch_index)
        
    train_rmse = sum(train_rmses) / len(train_rmses)

    classifier.eval()
    with torch.no_grad():
        for batch_index, data in enumerate(val_loader):   
            
            # load images, labels and random noise to device (GPU or CPU) 
            images, labels, cloud_mask, chip_id = data
            
            # images = images[:, [0,1,2,7,8,9], :, :, :].to(device, dtype=torch.float)
            images = images[:,[0,1,2,7],:,:,:].to(device, dtype=torch.float)
            labels = labels.to(device)
            
            # center crop
            images_cropped, labels_cropped = center_crop(images, labels, 224)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
    
            # extract feature vectors (tokens)
            features = encoder.forward_features(images_cropped) # 24, torch.Size([1, 1765, 256])
            # print(len(features), features[0].shape)
            
            # extract features from 6th, 12th, 18th, and last layers and remove CLS token before passing to segmentation head
            features = features[:,[5, 11, 17, 23],:,:] # torch.Size([20, 4, 197, 256])
            features = features[:,:,1:,:] # torch.Size([20, 4, 196, 256])
        
            # compute predictions
            pred = classifier(features)
            
            # compute per-image RMSEs
            batch_rmses = rmse_per_image(pred, labels_cropped)  # shape: [B]
            val_rmses.extend(batch_rmses.tolist())
            
            #logging.info(batch_index)
            
        val_rmse = sum(val_rmses) / len(val_rmses)
        logging.info(f"Epoch {epoch} - Training RMSE: {train_rmse:.4f} - Validation RMSE: {val_rmse:.4f} - Time: {time.time() - t0:.1f}s - LR: {scheduler.get_last_lr()[0]}")            
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(classifier.state_dict(), f"{filepath}/model-best.pt")
        logging.info('Best model saved with RMSE: %.2f'%(best_val_rmse))

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

val_rmses = []

classifier.eval()
with torch.no_grad():
    for batch_index, data in enumerate(val_loader):   
        
        # load images, labels and random noise to device (GPU or CPU) 
        images, labels, cloud_mask, chip_id = data
        
        # images = images[:, [0,1,2,7,8,9], :, :, :].to(device, dtype=torch.float)
        images = images[:,[0,1,2,7],:,:,:].to(device, dtype=torch.float)
        labels = labels.to(device)
        
        # center crop
        images_cropped, labels_cropped = center_crop(images, labels, 224)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped) # 24, torch.Size([1, 1765, 256])
        # print(len(features), features[0].shape)
        
        # extract features from 6th, 12th, 18th, and last layers and remove CLS token before passing to segmentation head
        features = features[:,[5, 11, 17, 23],:,:] # torch.Size([20, 4, 197, 256])
        features = features[:,:,1:,:] # torch.Size([20, 4, 196, 256])
    
        # compute predictions
        pred = classifier(features)
        
        # compute per-image RMSEs
        batch_rmses = rmse_per_image(pred, labels_cropped)  # shape: [B]
        val_rmses.extend(batch_rmses.tolist())
        
        #logging.info(batch_index)
        
    val_rmse = sum(val_rmses) / len(val_rmses)
    logging.info(f"Validation RMSE: {val_rmse:.4f}")   
  
###############################################################################
# EVALUATION - BEST MODEL
############################################################################### 
logging.info("Evaluation started.")

test_rmses = []

classifier.eval()
with torch.no_grad():
    for batch_index, data in enumerate(test_loader):   
        
        # load images, labels and random noise to device (GPU or CPU) 
        images, labels, cloud_mask, chip_id = data
        
        # images = images[:, [0,1,2,7,8,9], :, :, :].to(device, dtype=torch.float)
        images = images[:,[0,1,2,7],:,:,:].to(device, dtype=torch.float)
        labels = labels.to(device)
        
        # center crop
        images_cropped, labels_cropped = center_crop(images, labels, 224)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images_cropped) # 24, torch.Size([1, 1765, 256])
        # print(len(features), features[0].shape)
        
        # extract features from 6th, 12th, 18th, and last layers and remove CLS token before passing to segmentation head
        features = features[:,[5, 11, 17, 23],:,:] # torch.Size([20, 4, 197, 256])
        features = features[:,:,1:,:] # torch.Size([20, 4, 196, 256])
    
        # compute predictions
        pred = classifier(features)
        
        # compute per-image RMSEs
        batch_rmses = rmse_per_image(pred, labels_cropped)  # shape: [B]
        test_rmses.extend(batch_rmses.tolist())
        
        #logging.info(batch_index)
        
    test_rmse = sum(test_rmses) / len(test_rmses)
    logging.info(f"Test RMSE: {test_rmse:.4f}")   
    
logging.info(f"Training and evaluation complete. Logs saved to training.log in {filepath} directory.")






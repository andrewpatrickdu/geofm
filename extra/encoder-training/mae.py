#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:38:08 2025

python mae.py --num_train_samples 190000 --batch_size 256 --num_epochs 100 --encoder_embed_dim 256 --decoder_embed_dim 128 --checkpoint /data/ckpt-mae-256

python mae.py --num_train_samples 190000 --batch_size 256 --num_epochs 100 --encoder_embed_dim 512 --decoder_embed_dim 256 --checkpoint /data/ckpt-mae-512

@author: andrew
"""

import logging
import os 
import timeit
import time
import math

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from prithvi_mae import PrithviMAE

# Parse command-line arguments
import argparse
parser = argparse.ArgumentParser(description="Train a PrithviMAE model")
parser.add_argument('--num_train_samples', type=int, default=200000, help='Number of training samples')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--encoder_embed_dim', type=int, default=256, help='Encoder embedding dimension for the student')
parser.add_argument('--decoder_embed_dim', type=int, default=128, help='Decoder embedding dimension for the student')
parser.add_argument('--checkpoint', type=str, default='/home/andrew/GFM/ckpt-mae-256', help='Directory of training and evaluation results')
args = parser.parse_args()

# Create checkpoints folder
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
# Define number of samples for training and validation
num_train_samples = args.num_train_samples
num_val_samples = int(num_train_samples*0.05)

class HLSDataset(Dataset):
    def __init__(self, save_dir, start_idx, num_samples, transform=None):
        self.save_dir = save_dir
        self.file_names = sorted([f for f in os.listdir(save_dir) if f.endswith('.npy')])[start_idx:start_idx + num_samples]
        self.transform = transform
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.save_dir, self.file_names[idx])
        np_image = np.load(file_path)
        return np_image


# Image Transformations for PyTorch
transform = transforms.Compose([
    #transforms.Resize((224, 224)),  # Resize to 128x128
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),          # Convert to Tensor
])

batch_size = args.batch_size
train_dataset = HLSDataset(save_dir='/data/hls_dataset', start_idx=0, num_samples=num_train_samples, transform=transform)
val_dataset = HLSDataset(save_dir='/data/hls_dataset', start_idx=num_train_samples, num_samples=num_val_samples, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
logging.info(f"Loaded training dataset with {len(train_dataset)} samples.")
logging.info(f"Loaded validation dataset with {len(val_dataset)} samples.")

###############################################################################
# MODEL
###############################################################################
# Initialize model
model = PrithviMAE(in_chans = 4, embed_dim = args.encoder_embed_dim, decoder_embed_dim = args.decoder_embed_dim).to(device)
logging.info("Model initialized.")

# CHCKER: Display total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Model has {total_params:,} parameters.")

# # CHECKER - display trainable model parameters "
# for name, param in model.named_parameters():
#     if param.requires_grad == True:
#         # print(name, param.data)
#         print(name)

###############################################################################
# TRAINING PARAMETERS
###############################################################################  
# Define optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05
)

num_epochs = args.num_epochs
num_warmup_epochs = num_epochs*0.1 # 10% of total epochs

lr_start = 1e-6
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
# MAE
############################################################################### 
# Start timer
start = timeit.default_timer()
logging.info("Training started.")
logging.info(f"Training Parameters - Batch Size: {batch_size}, Epochs: {num_epochs}, Model embed_dim: {model.encoder.embed_dim}, Model decoder_embed_dim: {model.decoder.decoder_embed_dim}")

# Define lists to keep track of losses and learning rates 
train_losses = []
valid_losses = []
learning_rates = []

# Perform distillation 
for epoch in range(1, num_epochs + 1):

    # start timer
    t0 = time.time()

    # keep track of losses    
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for images in train_loader:
 
        # load image and random noise to device (GPU or CPU)
        images = images.to(device)
        random_noise = torch.rand(images.size(0), model.sequence_length-1).to(device)
        
        # select bands - [RGB, NIR]: bands 1,2,3,4
        images = images[:,[0,1,2,3],:,:]

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        # forward propagation and loss computation
        loss, pred, mask = model(images.unsqueeze(2), random_noise, mask_ratio=0.75)

        # backward propagation: compute gradient of the loss wrt model parameters
        loss.backward()

        # update the model parameters
        optimizer.step()

        # update training loss
        train_loss += loss.item() * images.size(0)
        
        # delete tensors
        del loss, pred, mask, random_noise, images

        # clear cache to prevent memory accumulation
        torch.cuda.empty_cache()

    # calculate average training loss
    train_loss = train_loss/len(train_loader.sampler)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        for images in valid_loader:
            
            # load image and random noise to device (GPU or CPU)
            images = images.to(device)
            random_noise = torch.rand(images.size(0), model.sequence_length-1).to(device)

            # select bands - [RGB, NIR]: bands 1,2,3,4
            images = images[:,[0,1,2,3],:,:]
            
            # forward propagation
            loss, pred, mask = model(images.unsqueeze(2), random_noise, mask_ratio=0.75)
    
            # update validation loss
            valid_loss += loss.item() * images.size(0)
            
            # delete tensors
            del loss, pred, mask, random_noise, images

            # clear cache to prevent memory accumulation
            torch.cuda.empty_cache()

    # calculate average training loss
    valid_loss = valid_loss/len(valid_loader.sampler)
    valid_losses.append(valid_loss)

    # update learning rate    
    scheduler.step()
    learning_rates.append(scheduler.get_last_lr()[0])
    
    logging.info(f"Epoch {epoch}: Training Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}, Learning Rate: {scheduler.get_last_lr()[0]:.12f}, Time: {time.time() - t0:.2f}s")

    # save model every 20 epochs
    if epoch%20 == 0:
        torch.save(model.state_dict(), f'{filepath}/model-{epoch}.ckpt')
        logging.info(f"Checkpoint saved at epoch {epoch}.")

# Stop timer
stop = timeit.default_timer()
total_time = stop - start
logging.info(f"Total training time: {int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s")

# Save the trained model
torch.save(model.state_dict(), f'{filepath}/model-final.pt')
logging.info("Final model saved.")

# Save losses and learning rate information
import pickle
with open(f'{filepath}/train_loss.pkl', 'wb') as file:
    pickle.dump(train_losses, file)

with open(f'{filepath}/validation_loss.pkl', 'wb') as file:
    pickle.dump(valid_losses, file)

# CHECKER
# with open('checkpoints/train_loss.pkl', 'rb') as file:
#     loaded_list = pickle.load(file)
# print(loaded_list)  

with open(f'{filepath}/learning_rates.pkl', 'wb') as file:
    pickle.dump(learning_rates, file)  
    
# Plot loss curves
import matplotlib.pyplot as plt
plt.figure(figsize=[8,6])
plt.plot(train_losses, 'b', label='Training loss')
plt.plot(valid_losses, 'r', label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.legend(frameon=False)
plt.savefig(f"{filepath}/loss_plot.png")

# Plot learning rate 
import matplotlib.pyplot as plt
plt.figure(figsize=[8,6])
plt.plot(learning_rates, 'b', label='Learning rate')
plt.xlabel("Epochs")
plt.ylabel("Learning rate")
plt.grid(color='blue', linestyle='--', linewidth=0.5)
plt.legend(frameon=False)
plt.savefig(f"{filepath}/learning_rate_plot.png")

logging.info(f"Training complete. Logs saved to training.log in {filepath} directory.")



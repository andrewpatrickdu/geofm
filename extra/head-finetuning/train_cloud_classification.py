#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that trains a cloud detector.

python train_cloud_classification.py --batch_size 64 --num_epochs 300 --encoder /data/Prithvi_EO_V2_300M.pt --encoder_embed_dim 1024 --checkpoint /data/ckpt-cloud-classification-baseline-1024

python train_cloud_classification.py --batch_size 128 --num_epochs 300 --encoder /data/ckpt-mae-512/model-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-cloud-classification-mae-512

python train_cloud_classification.py --batch_size 128 --num_epochs 300 --encoder /data/ckpt-mae-256/model-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-cloud-classification-mae-256

python train_cloud_classification.py --batch_size 128 --num_epochs 300 --encoder /data/ckpt-distillation-512/student-final.pt --encoder_embed_dim 512 --checkpoint /data/ckpt-cloud-classification-distillation-512

python train_cloud_classification.py --batch_size 128 --num_epochs 300 --encoder /data/ckpt-distillation-256/student-final.pt --encoder_embed_dim 256 --checkpoint /data/ckpt-cloud-classification-distillation-256

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
    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name,label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        
        image = np.load(img_path)
        # image = img.imread(img_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

img_dir = '/data/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/images'
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
# test_70 = pd.concat([cloudy[int(0.85*N):int(len(cloudy)*1.00)], not_cloudy[int(0.85*N):int(len(not_cloudy)*1.00)]])

# Image transformations
train_transform = transforms.Compose([
                                     transforms.ToTensor(),
                                     # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                     # AddGaussianNoise(0., 1.),
                                     # transforms.ToPILImage(),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
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
train_dataset_70 = S2CloudMaskCatalogue(training_70, img_dir, train_transform)
val_dataset_70 = S2CloudMaskCatalogue(validation_70, img_dir, valid_transform)
# test_data_70 = S2CloudMaskCatalogue(test_70, img_dir, test_transform)

train_loader_70 = DataLoader(dataset=train_dataset_70, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader_70 = DataLoader(dataset=val_dataset_70, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
logging.info(f"Loaded training dataset with {len(train_dataset_70)} samples.")
logging.info(f"Loaded validation dataset with {len(val_dataset_70)} samples.")

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
# CLASSIFIER MODEL
###############################################################################
import torch.nn as nn

class CloudClassifier(nn.Module):
    """
    A classification head with two fully connected layers.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_classes: int):
        super(CloudClassifier, self).__init__()
        
        # fully connected
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        
        # fc1
        x = self.fc1(x)
        x = self.relu(x)
        
        # fc2
        x = self.fc2(x)
        
        return x

classifier = CloudClassifier(input_dim=args.encoder_embed_dim, hidden_dim=512, num_classes=2).to(device)
logging.info("Classifier initialized.")

# CHCKER: Display total number of parameters
total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
logging.info(f"Classifier has {total_params:,} parameters.")

###############################################################################
# TRAINING PARAMETERS
###############################################################################  
# Define loss
weight = torch.tensor([2., 1.]).cuda()
criterion = nn.CrossEntropyLoss(
    weight=weight, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

# Define optimiser
optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-5)

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
# FINE-TUNING
############################################################################### 
# Start timer
start = timeit.default_timer()
logging.info("Training started.")
logging.info(f"Training Parameters - Batch Size: {batch_size}, Epochs: {num_epochs}, Encoder embed_dim: {encoder.embed_dim}, Classifier input_dim: {classifier.fc1.in_features}")

# Define lists to keep track of losses and accuracies
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
 
# Perform fine-tuning
for epoch in range(1, num_epochs + 1):

    # start timer
    t0 = time.time()

    # keep track of training loss and accuracy
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0
    valid_loss = 0.0
    valid_correct = 0.0
    valid_total = 0.0
    
    encoder.eval()
    classifier.train()
    for images, labels in train_loader_70:   

        # load image and random noise to device (GPU or CPU)     
        # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float)  
        images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float)  
        random_noise = torch.rand(images.size(0), encoder.sequence_length-1).to(device)
        
        # resize image
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # # CHECKER - plot image
        # img = images[1]
        # img = img.permute(1, 2, 0)
        # img = img.cpu()
        # img = img.numpy()
        # plt.imshow(img[:,:,[2,1,0]])
        # plt.show()

        # load ground truths to device (GPU or CPU)
        labels = labels.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # extract feature vectors (tokens)
        features = encoder.forward_features(images.unsqueeze(2)) # print(len(features), features[0].shape)
        features = features[:,-1,:,:]
        
        # extract the CLS token (first token)
        cls_token = features[:, 0, :]

        # compute logits
        logits = classifier(cls_token)

        # compute batch loss
        loss = criterion(logits, labels)

        # backward propagation: compute gradient of the loss wrt model parameters
        loss.backward()

        # update the model parameters
        optimizer.step()
        
        # update training loss
        train_loss += loss.item() * images.size(0)
        
        # update training accuracy
        _, predicted = torch.max(logits.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()        
        
        # delete tensors
        del loss, logits, cls_token, features, labels, images
        
        # clear cache to prevent memory accumulation
        torch.cuda.empty_cache()        

    # calculate average training loss and accuracy
    train_loss = train_loss/len(train_loader_70.sampler)
    train_losses.append(train_loss)
    train_acc = 100 * train_correct / train_total
    train_accuracies.append(train_acc)
        
    classifier.eval()
    with torch.no_grad():
        for images, labels in valid_loader_70:       
    
            # load image and random noise to device (GPU or CPU)     
            # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float) 
            images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float)  
            random_noise = torch.rand(images.size(0), encoder.sequence_length-1).to(device)
            
            # resize image
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
            # # CHECKER - plot image
            # img = images[1]
            # img = img.permute(1, 2, 0)
            # img = img.cpu()
            # img = img.numpy()
            # plt.imshow(img[:,:,[2,1,0]])
            # plt.show()
    
            # load ground truths to device (GPU or CPU)
            labels = labels.to(device)
    
            # extract feature vectors (tokens)
            features = encoder.forward_features(images.unsqueeze(2)) # print(len(features), features[0].shape)
            features = features[:,-1,:,:]
            
            # extract the CLS token (first token)
            cls_token = features[:, 0, :]
    
            # compute logits
            logits = classifier(cls_token)
    
            # compute batch loss
            loss = criterion(logits, labels)
    
            # update validation loss
            valid_loss += loss.item() * images.size(0)
    
            # update validation accuracy
            _, predicted = torch.max(logits.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()

    # calculate average validation losses and accuracy
    valid_loss = valid_loss/len(valid_loader_70.sampler)
    valid_losses.append(valid_loss)
    valid_acc = 100 * valid_correct / valid_total
    valid_accuracies.append(valid_acc)
    
    # update learning rate
    scheduler.step()
    
    logging.info(f"Epoch {epoch}: Training Loss: {train_loss:.6f}, Validation Loss: {valid_loss:.6f}, Training Accuracy: {train_acc:.2f}, Validation Accuracy: {valid_acc:.2f}, Learning Rate: {scheduler.get_last_lr()[0]:.12f}, Time: {time.time() - t0:.2f}s")

# Stop timer
stop = timeit.default_timer()
total_time = stop - start
logging.info(f"Total training time: {int(total_time // 3600)}h {int((total_time % 3600) // 60)}m {int(total_time % 60)}s")

# Save the trained model
torch.save(classifier.state_dict(), f'{filepath}/model-final.pt')
logging.info("Final model saved.")

# Save loss and accuracy information
import pickle
with open(f'{filepath}/train_loss.pkl', 'wb') as file:
    pickle.dump(train_losses, file)

with open(f'{filepath}/validation_loss.pkl', 'wb') as file:
    pickle.dump(valid_losses, file)

with open(f'{filepath}/train_accuracy.pkl', 'wb') as file:
    pickle.dump(train_accuracies, file)

with open(f'{filepath}/validation_accuracy.pkl', 'wb') as file:
    pickle.dump(valid_accuracies, file)

# Plot loss curves
plt.figure(figsize=[8,6])
plt.plot(train_losses, 'b', label='Training loss')
plt.plot(valid_losses, 'r', label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.legend(frameon=False)
plt.savefig(f"{filepath}/loss_plot.png")

# Plot accuracy curves
plt.figure(figsize=[8,6])
plt.plot(train_accuracies, 'b', label='Training Accuracy')
plt.plot(valid_accuracies, 'r', label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(color='green', linestyle='--', linewidth=0.5)
plt.legend(frameon=False)
plt.savefig(f"{filepath}/accuracy_plot.png")

###############################################################################
# EVALUATION
############################################################################### 
total_predictions = []
total_labels = []

logging.info("Evaluation started.")
classifier.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader_70:       

        # load image and random noise to device (GPU or CPU)     
        # images = images[:,[1,2,3,8,11,12],:,:].to(device, dtype=torch.float)  
        images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float)  
        random_noise = torch.rand(images.size(0), encoder.sequence_length-1).to(device)
        
        # resize image
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # # CHECKER - plot image
        # img = images[1]
        # img = img.permute(1, 2, 0)
        # img = img.cpu()
        # img = img.numpy()
        # plt.imshow(img[:,:,[2,1,0]])
        # plt.show()

        # load ground truths to device (GPU or CPU)
        labels = labels.to(device)

        # extract feature vectors (tokens)
        features = encoder.forward_features(images.unsqueeze(2)) # print(len(features), features[0].shape)
        features = features[:,-1,:,:]
        
        # extract the CLS token (first token)
        cls_token = features[:, 0, :]

        # compute logits
        logits = classifier(cls_token)

        # calcualte testing accuracy
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_predictions.append(predicted)
        total_labels.append(labels)

logging.info('Test accuracy: {}%'.format(100 * correct / total))

# Calculate the confusion matrix - 1: Cloudy and 0: Not Cloudy"
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

# Flatten lists
prediction = [item for sublist in total_predictions for item in sublist]
truth = [item for sublist in total_labels for item in sublist]

# convert lists to tensor
prediction = torch.FloatTensor(prediction)
truth = torch.FloatTensor(truth)

# calculate the confusion matrix
confusion_matrix = confusion(prediction, truth)

# false postive rate of test set
fp = 100 * confusion_matrix[1] / prediction.shape[0]
logging.info('False positive rate: {}%'.format(fp))

logging.info(f"Training and evaluation complete. Logs saved to training.log in {filepath} directory.")




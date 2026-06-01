#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that evaluates the cloud detector.

# Prithvi v2 (with embedding dimension of 256) pretrained via dual-MAE distillation
python evaluate_cloud_classification.py --encoder /home/andrew/github/GFM/ckpt-distillation-256/student-final.pt  --encoder_embed_dim 256 --classifier /home/andrew/github/GFM/ckpt-cloud-classification-distillation-256/model-final.pt

### EXPECTED OUTPUT ###
Test accuracy: 87.93103448275862%
False positive rate: 3.4482758620689653%
Test F1: 87.27272727272727%


# Prithvi v2 (with embedding dimension of 1024) - Original model
python evaluate_cloud_classification.py --encoder /home/andrew/github/GFM/Prithvi_EO_V2_300M.pt  --encoder_embed_dim 1024 --classifier /home/andrew/github/GFM/ckpt-cloud-classification-baseline-1024/model-final.pt

### EXPECTED OUTPUT ###
Test accuracy: 87.93103448275862%
False positive rate: 3.793103448275862%
Test F1: 87.36462093862815%



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

from sklearn.metrics import f1_score

# set random seeds for reproducibility
seed = 0 # must be set to 0 since encoder (frozen) + classifier was trained on seed = 0
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

img_dir = '/home/andrew/cloud-detector/datasets/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/images'
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

batch_size = 1
train_dataset_70 = S2CloudMaskCatalogue(training_70, img_dir, train_transform)
val_dataset_70 = S2CloudMaskCatalogue(validation_70, img_dir, valid_transform)
# test_data_70 = S2CloudMaskCatalogue(test_70, img_dir, test_transform)

#train_loader_70 = DataLoader(dataset=train_dataset_70, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader_70 = DataLoader(dataset=val_dataset_70, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
print(f"Loaded training dataset with {len(train_dataset_70)} samples.")
print(f"Loaded validation dataset with {len(val_dataset_70)} samples.")

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
print("Classifier initialized.")

# CHCKER: Display total number of parameters
total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
print(f"Classifier has {total_params:,} parameters.")

# Load checkpoint (pretrained weights)
checkpoint = args.classifier
state_dict = torch.load(checkpoint, map_location=device)

classifier.load_state_dict(state_dict)
print(f"Loaded checkpoint from {checkpoint}.")


###############################################################################
# EVALUATION
############################################################################### 
total_predictions = []
total_labels = []

i = 1

print("Evaluation started.")
encoder.eval()
classifier.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader_70:       

        # load image and random noise to device (GPU or CPU)     
        images = images[:,[1,2,3,8],:,:].to(device, dtype=torch.float)  
        
        # resize image
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = images.unsqueeze(2)

        # load ground truths to device (GPU or CPU)
        labels = labels.to(device)

        # extract feature vectors (tokens)
        features = encoder.forward_features(images) # print(len(features), features[0].shape) # torch.Size([1, 24, 197, 256])
        features = features[:,-1,:,:]
        
        # extract the CLS token (first token)
        cls_token = features[:,0,:]

        # compute logits
        logits = classifier(cls_token)

        # calcualte testing accuracy
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        total_predictions.append(predicted.detach().cpu())
        total_labels.append(labels.detach().cpu())

test_f1 = f1_score(torch.cat(total_labels), torch.cat(total_predictions), average='binary')
print('Test accuracy: {}%'.format(100 * correct / total))

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
print('False positive rate: {}%'.format(fp))
print('Test F1: {}%'.format(100 * test_f1))




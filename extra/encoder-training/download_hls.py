#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:19:50 2025

@author: andrew
"""

import ee
import datetime
import numpy as np
import random
import requests
import os 
import csv
import io

# Directory to save images
save_dir = "hls_dataset"
os.makedirs(save_dir, exist_ok=True)

# File to store collected sample locations
CSV_FILE = save_dir + "/" + "sampled_locations.csv"

# If the file doesn't exist, create it with headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Latitude", "Longitude"])

# Authenticate and Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-andrewdu468')
print(ee.String('Hello from GEE!').getInfo())

def get_hls_patch():
    """Fetches a random HLS image patch from a random location.
       Keeps looping until a valid image is found (infinite loop).
    """
    while True:  # Infinite loop until a valid image is found
        lat = random.uniform(-60, 75)  # Avoid extreme poles
        lon = random.uniform(-180, 180)
        aoi = ee.Geometry.Point([lon, lat])

        # Randomly select either Landsat or Sentinel
        dataset_choice = random.choice(["Landsat 8/9", "Sentinel 2"])
        # dataset_choice = "Landsat 8/9"
        # dataset_choice = "Sentinel 2"

        if dataset_choice == "Landsat 8/9":
            dataset = ee.ImageCollection("projects/earthengine-public/assets/NASA/HLS/HLSL30/v002") \
                .filterBounds(aoi) \
                .filterDate("2015-11-28", "2025-02-13") \
                .filter(ee.Filter.lt('CLOUD_COVERAGE', 20))
            bands = ["B2", "B3", "B4", "B5", "B6", "B7"]  # Landsat 8/9 bands
        else:      
            dataset = ee.ImageCollection("projects/earthengine-public/assets/NASA/HLS/HLSS30/v002") \
                .filterBounds(aoi) \
                .filterDate("2015-11-28", "2025-02-13") \
                .filter(ee.Filter.lt('CLOUD_COVERAGE', 20))
            bands = ["B2", "B3", "B4", "B8A", "B11", "B12"]  # Sentinel-2 bands

        # Count total images
        total_images = dataset.size().getInfo()
        if total_images == 0:
            print(f"❌ No {dataset_choice} images found at {lat}, {lon}. Retrying...")
            continue  # Restart loop if no images are found

        # Select a random image index
        random_index = random.randint(0, total_images - 1)
        img = dataset.toList(total_images).get(random_index)
        img = ee.Image(img)  # Convert to an ee.Image

        # Extract image acquisition date (convert from milliseconds to YYYY-MM-DD)
        date_millis = img.get("system:time_start").getInfo()
        date_str = datetime.datetime.utcfromtimestamp(date_millis / 1000).strftime('%Y-%m-%d')

        # Extract Cloud Coverage
        cloud_coverage = img.get("CLOUD_COVERAGE").getInfo()

        print(f"✅ Selected {dataset_choice} image at {lat}, {lon} on {date_str} with {cloud_coverage:.2f}% cloud cover")
        
        # Add Fmask band 
        fmask_band = "Fmask"
        all_bands = bands + [fmask_band]
        img_with_fmask = img.select(all_bands)
        
        # Request image download
        url = img_with_fmask.getDownloadURL({
            'scale': 30,
            'region': aoi.buffer(3400).bounds(),
            'format': 'NPY'})
        print(f"📸 Downloading image from: {url}")

        response = requests.get(url)
        np_image = np.load(io.BytesIO(response.content))
        
        # Centre crop image
        def center_crop(image_array, target_size=224):
            H, W = image_array.shape[-2:]  # Get last two dimensions
            start_x = (W - target_size) // 2
            start_y = (H - target_size) // 2
            return image_array[..., start_y:start_y + target_size, start_x:start_x + target_size]
        
        np_image = center_crop(np_image, 224)
        
        #######################################################################
        # # Extract Fmask 
        # fmask = np_image[fmask_band]

        # # Check individual bits
        # cirrus = np.bitwise_and(fmask, 1 << 0) > 0
        # cloud = np.bitwise_and(fmask, 1 << 1) > 0
        # adjacent_cloud_shadow = np.bitwise_and(fmask, 1 << 2) > 0
        # cloud_shadow = np.bitwise_and(fmask, 1 << 3) > 0
        # snow_ice = np.bitwise_and(fmask, 1 << 4) > 0
        # water = np.bitwise_and(fmask, 1 << 5) > 0
        # aerosol_level = np.right_shift(fmask, 6) & 3  # Bits 6 and 7 (aerosol)

        # # Total cloud pixels (cloud + cloud shadow)
        # cloudy_pixels = cloud | cloud_shadow
        # cloudy_pixel_ratio = np.mean(cloudy_pixels)
        # print(f"Cloudy pixels (Fmask): {cloudy_pixel_ratio:.2%}")
        #######################################################################

        # Extract band images
        np_image = np.stack([np_image[band] for band in bands], axis=0)

        # If more than 1% missing values (-3.2768 or NaNs), skip and retry
        missing_pixels = (np_image == -3.2768) | np.isnan(np_image)
        if np.mean(missing_pixels) > 0.01:
            print(f"❌ Skipping image due to missing pixels: {np.mean(missing_pixels):.2%}")
            continue
                
        #######################################################################
        # CHECKER:
        # print('Dimension of image:', np_image.shape)
        # print('Minimum pixel value:', np.min(np_image))
        # print('Maximum pixel value:', np.max(np_image))
        # print('Missing pixel value (NaN):', np.isnan(np_image).sum())
        # print('Missing pixel value (-3.2768):', (np_image == -3.2768).sum()/np_image.size)
        
        # import matplotlib.pyplot as plt
        # img_plot = np.transpose(np_image[[2,1,0],:,:], (1, 2, 0))
        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # axes[0].imshow(img_plot)
        # axes[0].set_title("Original")
        # axes[0].axis('off')
        # axes[1].imshow(img_plot*2.5)
        # axes[1].set_title("Original (2.5x)")
        # axes[1].axis('off')
        # plt.tight_layout()
        # plt.show()
        #######################################################################
        
        # from IPython import embed; embed()

        # Clip values between 0 and 1
        np_image = np.clip(np_image, 0, 1)
        
        np_image = np_image.astype(np.float32)
        
        # Save the sampled location
        with open(CSV_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([lat, lon])

        return np_image 

num_samples = 200000

for idx in range(num_samples):
    try:
        print(f"Downloading image {idx+1}/{num_samples}")
        np_image = get_hls_patch()
        np.save(os.path.join(save_dir, f"sample_{idx+1:06d}.npy"), np_image)
    except Exception as e:
        print(f"❌ Error downloading image {idx+1}: {e}. Retrying...")
        continue

print("✅ Completed downloading 200,000 samples!")






















# you should be able to run this script from the root directory of the project and produce tiles as specified
# see commented out section out bottom to verify results

import os
import numpy as np
import rasterio
import tensorflow as tf

def split_pad_image(image, target_height, target_width):
    bands, height, width = image.shape
    padded_height = int(np.ceil(height / target_height) * target_height)
    padded_width = int(np.ceil(width / target_width) * target_width)
    
    padded_image = np.pad(image, ((0, 0), (0, padded_height - height), (0, padded_width - width)), mode='constant')
    
    num_tiles_h = padded_height // target_height
    num_tiles_w = padded_width // target_width
    
    tiles = []
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            tile = padded_image[:, i*target_height:(i+1)*target_height, j*target_width:(j+1)*target_width]
            tiles.append(tile)
    return tiles


def save_tiles(tiles, output_dir, basename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, tile in enumerate(tiles):
        tile_path = os.path.join(output_dir, f"{basename}_tile_{i}.tif")
        bands, height, width = tile.shape
        with rasterio.open(tile_path, 'w', driver='GTiff', height=height, width=width, count=bands, dtype=tile.dtype) as dst:
            for b in range(bands):
                dst.write(tile[b], b+1)


target_tile_height = 256
target_tile_width = 256

# Set the paths to the respective directories
amazon_training_image_dir = '../data/download/AMAZON/Training/image/'
amazon_training_label_dir = '../data/download/AMAZON/Training/label/'

amazon_validation_image_dir = '../data/download/AMAZON/Validation/images/'
amazon_validation_label_dir = '../data/download/AMAZON/Validation/masks/'

amazon_test_image_dir = '../data/download/AMAZON/Test/image/'
amazon_test_label_dir = '../data/download/AMAZON/Test/mask/'

# Create lists of image and label file paths for all .tif
amazon_training_image_paths = [os.path.join(amazon_training_image_dir, filename) for filename in os.listdir(amazon_training_image_dir) if filename.endswith('.tif')]
amazon_training_label_paths = [os.path.join(amazon_training_label_dir, filename) for filename in os.listdir(amazon_training_label_dir) if filename.endswith('.tif')]

amazon_validation_image_paths = [os.path.join(amazon_validation_image_dir, filename) for filename in os.listdir(amazon_validation_image_dir) if filename.endswith('.tif')]
amazon_validation_label_paths = [os.path.join(amazon_validation_label_dir, filename) for filename in os.listdir(amazon_validation_label_dir) if filename.endswith('.tif')]

amazon_test_image_paths = [os.path.join(amazon_test_image_dir, filename) for filename in os.listdir(amazon_test_image_dir) if filename.endswith('.tif')]
amazon_test_label_paths = [os.path.join(amazon_test_label_dir, filename) for filename in os.listdir(amazon_test_label_dir) if filename.endswith('.tif')]

# --- Training ---
for image_path, label_path in zip(amazon_training_image_paths, amazon_training_label_paths):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    with rasterio.open(image_path) as src:
        image = src.read()
    with rasterio.open(label_path) as src:
        label = src.read()
    
    image_tiles = split_pad_image(image, target_tile_height, target_tile_width)
    label_tiles = split_pad_image(label, target_tile_height, target_tile_width)
    
    save_tiles(image_tiles, amazon_training_image_dir.replace('download', 'edit'), basename)
    save_tiles(label_tiles, amazon_training_label_dir.replace('download', 'edit'), basename)

# --- Validation ---
for image_path, label_path in zip(amazon_validation_image_paths, amazon_validation_label_paths):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    with rasterio.open(image_path) as src:
        image = src.read()
    with rasterio.open(label_path) as src:
        label = src.read()
    
    image_tiles = split_pad_image(image, target_tile_height, target_tile_width)
    label_tiles = split_pad_image(label, target_tile_height, target_tile_width)
    
    save_tiles(image_tiles, amazon_validation_image_dir.replace('download', 'edit'), basename)
    save_tiles(label_tiles, amazon_validation_label_dir.replace('download', 'edit'), basename)

# --- Test ---
for image_path, label_path in zip(amazon_test_image_paths, amazon_test_label_paths):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    
    with rasterio.open(image_path) as src:
        image = src.read()
    with rasterio.open(label_path) as src:
        label = src.read()
    
    image_tiles = split_pad_image(image, target_tile_height, target_tile_width)
    label_tiles = split_pad_image(label, target_tile_height, target_tile_width)
    
    save_tiles(image_tiles, amazon_test_image_dir.replace('download', 'edit'), basename)
    save_tiles(label_tiles, amazon_test_label_dir.replace('download', 'edit'), basename)


# Verify

# from visualize_geotiff import visualize_geotiff
# sample_path_original = amazon_training_image_dir
# sample_files_original = os.listdir(sample_path_original)
# sample_files_original.sort()
# sample_path = amazon_training_image_dir.replace('download', 'edit')
# sample_files = os.listdir(sample_path)
# sample_files.sort()
# sample_files

# for i in range(4):
#     visualize_geotiff(sample_path + sample_files[i])
# visualize_geotiff(sample_path_original + sample_files_original[0])


# Check

# import os
# import rasterio
# import pandas as pd

# def check_dimensions(folder_path):
#     file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.tif')]
#     for file_path in file_paths:
#         with rasterio.open(file_path) as src:
#             width, height = src.width, src.height
#             channels = src.count
#         pd.DataFrame([[file_path, f"({width}, {height})", channels]], columns=['File', 'Dimensions', 'Channels']).to_csv("../data/edit/generated_tile_shape_diagnostics.csv", mode='a', header=False, index=False)

# # Set folder paths
# folders = [
#     amazon_training_image_dir.replace('download', 'edit'),
#     amazon_training_label_dir.replace('download', 'edit'),
#     amazon_validation_image_dir.replace('download', 'edit'),
#     amazon_validation_label_dir.replace('download', 'edit'),
#     amazon_test_image_dir.replace('download', 'edit'),
#     amazon_test_label_dir.replace('download', 'edit')
# ]

# # Check dimensions for each folder
# pd.DataFrame(columns=['File', 'Dimensions', 'Channels']).to_csv("../data/edit/generated_tile_shape_diagnostics.csv", index=False)
# for folder in folders:
#     print(f"Checking dimensions for files in folder: {folder}")
#     check_dimensions(folder)
#     print("\n")

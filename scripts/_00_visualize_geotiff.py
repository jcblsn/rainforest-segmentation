# visualize geotiff images and masks

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

def visualize_geotiff(image_path, mask_path=None, save_path=None):
    
    # read image data
    with rasterio.open(image_path) as src:
        image_data = src.read().astype(np.float32)
    
    # stack RGB bands for visualization (check image_data.shape if this isn't clear)
    # need to normalize each band to 0-255 (this is just for eye-checking so the formula isn't important)
        image_rgb = np.stack(
        (
            (image_data[0] - np.min(image_data[0])) * 255.0 / (np.max(image_data[0]) - np.min(image_data[0])),
            (image_data[1] - np.min(image_data[1])) * 255.0 / (np.max(image_data[1]) - np.min(image_data[1])),
            (image_data[2] - np.min(image_data[2])) * 255.0 / (np.max(image_data[2]) - np.min(image_data[2]))
        ),
        axis=-1
    ).astype(np.uint8)
    
    # read mask if provided
    if mask_path:
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
        
        # plot both
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(image_rgb)
        plt.title("Image")
        plt.axis("off")
        
        plt.subplot(122)
        plt.imshow(mask_data, cmap="gray")
        plt.title("Mask")
        plt.axis("off")
        
    # plot only image if no mask
    else:
        plt.figure(figsize=(6, 6))
        plt.imshow(image_rgb)
        plt.title("Image")
        plt.axis("off")
    
    # save to disk if save_to is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


image_path_amz = "../data/download/AMAZON/Training/image/"
label_path_amz = "../data/download/AMAZON/Training/label/"

image_path_atl = "../data/download/ATLANTIC FOREST/Training/image/"
label_path_atl = "../data/download/ATLANTIC FOREST/Training/label/"

images_amz = os.listdir(image_path_amz)
labels_amz = os.listdir(label_path_amz)

images_atl = os.listdir(image_path_atl)
labels_atl = os.listdir(label_path_atl)

i = 0

# visualize in jupyter notebook
# visualize_geotiff(image_path_amz+images_amz[i], label_path_amz+labels_amz[i])
# visualize_geotiff(image_path_amz+images_amz[i])

# visualize_geotiff(image_path_atl+images_atl[i], label_path_atl+labels_atl[i])
# visualize_geotiff(image_path_atl+images_atl[i])

# save to disk
# save_to = "../plot/rgb-samples/amazon/"
# if not os.path.exists(save_to):
#     os.makedirs(save_to, exist_ok=True)

# for i in range(19):
#     visualize_geotiff(image_path_amz+images_amz[i], label_path_amz+labels_amz[i], save_path=save_to+images_amz[i].split(".")[0]+".png")

# # save to disk
# save_to = "../plot/rgb-samples/atlantic/"
# if not os.path.exists(save_to):
#     os.makedirs(save_to, exist_ok=True)

# for i in range(19):
#     visualize_geotiff(image_path_atl+images_atl[i], label_path_atl+labels_atl[i], save_path=save_to+images_atl[i].split(".")[0]+".png")
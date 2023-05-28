# based on: https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statistics import mean
import torch.nn as nn
from torch.nn.functional import threshold, normalize
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.optim.lr_scheduler import StepLR

# load dataset paths
cwd = os.path.dirname(__file__)+'/'
relative_path = '../data/edit/'

amazon_training_image_dir = cwd+relative_path+'AMAZON/Training/image/'
amazon_training_label_dir = cwd+relative_path+'AMAZON/Training/label/'

amazon_validation_image_dir = cwd+relative_path+'AMAZON/Validation/images/'
amazon_validation_label_dir = cwd+relative_path+'AMAZON/Validation/masks/'

amazon_test_image_dir = cwd+relative_path+'AMAZON/Test/image/'
amazon_test_label_dir = cwd+relative_path+'AMAZON/Test/mask/'

# Define a function to load images and labels from directory
def load_images_and_labels(image_dir, label_dir):
    image_paths = sorted([os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.tif')])
    label_paths = sorted([os.path.join(label_dir, filename) for filename in os.listdir(label_dir) if filename.endswith('.tif')])
    return image_paths, label_paths

# Load the images and labels
train_image_paths, train_label_paths = load_images_and_labels(amazon_training_image_dir, amazon_training_label_dir)
val_image_paths, val_label_paths = load_images_and_labels(amazon_validation_image_dir, amazon_validation_label_dir)
test_image_paths, test_label_paths = load_images_and_labels(amazon_test_image_dir, amazon_test_label_dir)

def preprocess_images_and_labels(image_paths, label_paths):
    preprocessed_data = []
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    for img_path, lbl_path in zip(image_paths, label_paths):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        resized_image = transform.apply_image(image)
        resized_label = transform.apply_image(label)
        input_image_torch = torch.as_tensor(resized_image, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = sam_model.preprocess(transformed_image)

        preprocessed_data.append({
            'image': input_image,
            'input_size': tuple(transformed_image.shape[-2:]),
            'original_image_size': image.shape[:2],
            'label': torch.tensor(resized_label, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        })

    return preprocessed_data

def sample_points_from_label(label, num_points=5):
    """
    Sample 5 labeled points that we'll use to prompt the model.
    Want coords, labels = points
    """
    label_np = label.squeeze(0).squeeze(0).cpu().numpy()

    sample_x = np.random.randint(0, label_np.shape[0], num_points)
    sample_y = np.random.randint(0, label_np.shape[1], num_points)
    sample_points = np.stack([sample_x, sample_y], axis=1)
    sample_labels = label_np[sample_x, sample_y]
    sampled_point_labels = label_np[sample_x, sample_y]
    labeled_sample_points = [(point, label) for point, label in zip(sample_points, sampled_point_labels)]

    return sample_points, sample_labels

def step(model, data, optimizer, loss_fn):
    image = data['image']
    label = data['label']

    # plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy()[:,:,0])
    # plt.imshow(label.squeeze(0).squeeze(0).cpu().numpy())

    # Image encoding
    with torch.no_grad():

        image_embedding = sam_model.image_encoder(image)

        # Visualize image embedding
        # plot_image_embedding = image_embedding.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # plt.imshow(plot_image_embedding[:,:,2])

        prompt_box = (0, 0, data['input_size'][1], data['input_size'][0])
        box_torch = torch.tensor(prompt_box, dtype=torch.float).unsqueeze(0)

        # prompt_points, prompt_point_labels = sample_points_from_label(label)
        # # from [(array([795, 325]), 0.0),...] to tensor
        # prompt_points_torch = torch.tensor(prompt_points, dtype=torch.float, device = device).unsqueeze(0)
        # prompt_point_labels_torch = torch.tensor(prompt_point_labels, dtype=torch.float, device = device).unsqueeze(0)

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=label
        )

        # Visualize dense_embedding: torch.Size([1, 256, 64, 64])
        # plot_dense_embedding = dense_embeddings.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # plt.imshow(plot_dense_embedding[:,:,0])

    # Mask decoding
    dense_embeddings_resized = nn.functional.interpolate(dense_embeddings, size=image_embedding.shape[2:], mode='bilinear', align_corners=False)
    low_res_masks, _ = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings_resized,
        multimask_output=False
    )

    upscaled_masks = sam_model.postprocess_masks(low_res_masks, data['input_size'], data['original_image_size'])
    binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
    
    # plot_mask = binary_mask.squeeze(0).squeeze(0).detach().numpy()
    # plt.imshow(plot_mask)

    label_resized = torch.nn.functional.interpolate(label, size=(binary_mask.shape[2], binary_mask.shape[3]), mode='nearest')

    loss = loss_fn(binary_mask, label_resized)
    return loss

def train_step(model, data, optimizer, loss_fn):
    
    loss = step(model, data, optimizer, loss_fn)

    # Update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def validation_step(model, data, loss_fn):
    
    with torch.no_grad():
        loss = step(model, data, None, loss_fn)

    return loss.item()


# Set up model, optimizer, and loss function
sam_checkpoint = "/Users/jacobeliason/Documents/Files/Code/Resources/segment-anything-checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 1e-3
weight_decay = 0
num_epochs = 10

sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device)
sam_model.train()

# Adjust img_size according to the target input image size (after init model)
train_data = preprocess_images_and_labels(train_image_paths, train_label_paths)
val_data = preprocess_images_and_labels(val_image_paths, val_label_paths)
test_data = preprocess_images_and_labels(test_image_paths, test_label_paths)

optimizer = torch.optim.SGD(sam_model.mask_decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = torch.nn.MSELoss()

idx = 400

train_loss = []
val_loss = []

for epoch in range(num_epochs):
    epoch_train_loss = []
    epoch_val_loss = []

    for data in train_data[:idx]:
        loss = train_step(sam_model, data, optimizer, loss_fn)
        epoch_train_loss.append(loss)
        train_loss.append(loss)

        # Print training loss every 10 batches
        if len(train_loss) % 5 == 0:
            print(f'Epoch: {epoch} | Batch: {len(train_loss)} | Iteration loss: {round(np.mean(train_loss),4)}')

    for data in val_data[:idx]:
        loss = validation_step(sam_model, data, loss_fn)
        epoch_val_loss.append(loss)
        val_loss.append(loss)

    print(f'Epoch: {epoch+1}: Training loss: {round(np.mean(epoch_train_loss),4)} | Validation loss: {round(np.mean(epoch_val_loss),4)}')


import pandas as pd
# pd.DataFrame(train_loss).to_csv('train_loss.csv')
# pd.DataFrame(val_loss).to_csv('val_loss.csv')

# read
train_loss = pd.read_csv('train_loss.csv', index_col=0)
val_loss = pd.read_csv('val_loss.csv', index_col=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_loss[:400], label='train')
ax.plot(val_loss[:400], label='val')
ax.legend()
plt.show()
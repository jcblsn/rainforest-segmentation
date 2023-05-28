import _02a_process_images

import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd

# -------------------- augmented --------------------

batches = 16
input_width = 256
shuffled = True
augmented = {
    "flip_left_right": 0,
    "flip_up_down": 0,
    "gaussian_blur": 0.2,
    "random_noise": 0.0,
    "random_brightness": 0.5,
    "random_contrast": 0.5
}

train_dataset, val_dataset, test_dataset, amazon_training_image_paths, amazon_training_label_paths, amazon_validation_image_paths, amazon_validation_label_paths, amazon_test_image_paths, amazon_test_label_paths = \
    _02a_process_images.process_data(
        batches = batches, 
        shuffle = shuffled,
        augmentation_settings = augmented, 
        image_width = input_width,
        relative_path = '../data/edit/'
    )

save_path = '../data/tf/augmented'
if not os.path.exists(save_path):
    os.makedirs(save_path)

train_path = os.path.join(save_path, 'train')
val_path = os.path.join(save_path, 'val')
test_path = os.path.join(save_path, 'test')

tf.data.experimental.save(train_dataset, train_path)
tf.data.experimental.save(val_dataset, val_path)
tf.data.experimental.save(test_dataset, test_path)

# --------------------- not augmented ---------------------

batches = 16
input_width = 256
shuffled = True
augmented = {
    "flip_left_right": 0,
    "flip_up_down": 0,
    "gaussian_blur": 0,
    "random_noise": 0,
    "random_brightness": 0,
    "random_contrast": 0
}

train_dataset, val_dataset, test_dataset, amazon_training_image_paths, amazon_training_label_paths, amazon_validation_image_paths, amazon_validation_label_paths, amazon_test_image_paths, amazon_test_label_paths = \
    _02a_process_images.process_data(
        batches = batches, 
        shuffle = shuffled,
        augmentation_settings = augmented, 
        image_width = input_width,
        relative_path = '../data/edit/'
    )

save_path = '../data/tf/not_augmented'
if not os.path.exists(save_path):
    os.makedirs(save_path)

train_path = os.path.join(save_path, 'train')
val_path = os.path.join(save_path, 'val')
test_path = os.path.join(save_path, 'test')

tf.data.experimental.save(train_dataset, train_path)
tf.data.experimental.save(val_dataset, val_path)
tf.data.experimental.save(test_dataset, test_path)


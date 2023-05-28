import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import rasterio

def process_data(
        batches = 16, 
        shuffle = True,
        augmentation_settings = None,
        image_width = 256,
        relative_path = '../data/edit/'
        ):

    """
    A function to process data by creating training, validation, and test datasets.
    note that:
     - we assume square images
     - there are outstanding issues with the augmentation function; feedback is welcome
    :param batches: Number of batches for the dataset
    :param shuffle: Whether to shuffle the dataset or not
    :param augmentation_settings: Dictionary of augmentation settings
    :param image_width: The width of the images
    :param relative_path: The path to the data directory within the root directory
    :return: train_dataset, val_dataset, test_dataset, and paths for images and labels
    """

    def read_image(file_path):
        file_path = file_path.decode('utf-8')
        with rasterio.open(file_path) as src:
            img_array = src.read()
        img_array = np.transpose(img_array, (1, 2, 0))
        return img_array.astype(np.float32)


    def _parse_function(image_path, label_path):
        image = tf.py_function(lambda img_path: read_image(img_path.numpy()), [image_path], tf.float32)
        label = tf.py_function(lambda lbl_path: read_image(lbl_path.numpy()), [label_path], tf.float32)
        return image, label

    def apply_augmentation(image, label, augmentation_tensor):

        if augmentation_settings is None:
            return image, label

        flip_left_right_prob, flip_up_down_prob, gaussian_blur_prob, random_noise_prob, random_brightness_prob, random_contrast_prob = tf.unstack(augmentation_tensor)
        
        if False:#flip_left_right_prob > tf.random.uniform((), maxval=1):
            seed1 = tf.random.uniform((), maxval=10000, dtype=tf.int32)
            image = tf.image.random_flip_left_right(image, seed=seed1)
            label = tf.image.random_flip_left_right(label, seed=seed1)

        if False:#flip_up_down_prob > tf.random.uniform((), maxval=1):
            seed2 = tf.random.uniform((), maxval=10000, dtype=tf.int32)
            image = tf.image.random_flip_up_down(image, seed=seed2)
            label = tf.image.random_flip_up_down(label, seed=seed2)

        if gaussian_blur_prob > tf.random.uniform((), maxval=1):
            image = tfa.image.gaussian_filter2d(image, sigma = 0.8)

        if False:#random_noise_prob > tf.random.uniform((), maxval=1):
            image = tf.clip_by_value(image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1.0), 0, 255)

        if random_brightness_prob > tf.random.uniform((), maxval=1):
            image = tf.image.adjust_brightness(image, delta=0.1)
            
        if random_contrast_prob > tf.random.uniform((), maxval=1):
            image = tf.image.adjust_contrast(image, contrast_factor=0.1)

        image = tf.cast(image, dtype=tf.float32)
        label = tf.cast(label, dtype=tf.float32)

        return image, label


    # def apply_augmentation(image, label, augmentation_tensor):
        
    #     augmentation_settings = {}
    #     augmentation_settings = dict(zip(augmentation_settings.keys(), augmentation_tensor.numpy().tolist()))


    #     if augmentation_settings is None:
    #         return image, label

    #     if augmentation_settings.get("flip_left_right", 0) > tf.random.uniform(()):
    #         seed = tf.random.uniform(())
    #         image = tf.image.random_flip_left_right(image, seed=seed)
    #         label = tf.image.random_flip_left_right(label, seed=seed)
            
    #     if augmentation_settings.get("flip_up_down", 0) > tf.random.uniform(()):
    #         seed = tf.random.uniform(())
    #         image = tf.image.random_flip_up_down(image, seed=seed)
    #         label = tf.image.random_flip_up_down(label, seed=seed)

    #     if augmentation_settings.get("gaussian_blur", 0) > tf.random.uniform(()):
    #         image = tfa.image.gaussian_filter2d(image)

    #     if augmentation_settings.get("random_noise", 0) > tf.random.uniform(()):
    #         image = tf.clip_by_value(image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05*255), 0, 255)

    #     if augmentation_settings.get("random_brightness", 0) > tf.random.uniform(()):
    #         image = tf.image.random_brightness(image, 0.2)

    #     if augmentation_settings.get("random_contrast", 0) > tf.random.uniform(()):
    #         image = tf.image.random_contrast(image, lower=0.5, upper=2.0)

    #     image = tf.cast(image, dtype=tf.float32)
    #     label = tf.cast(label, dtype=tf.float32)

    #     return image, label

    # def preprocess_data(image, label, augmentation_settings):
    #     if augmentation_settings:
    #         # image, label = tf.numpy_function(apply_augmentation, [image, label, augmentation_settings], [tf.float32, tf.float32])
    #         augmentation_tensor = tf.convert_to_tensor(list(augmentation_settings.values()), dtype=tf.float32)
    #         image, label = tf.numpy_function(apply_augmentation, [image, label, augmentation_tensor], [tf.float32, tf.float32])

    #         image.set_shape((image_width, image_width, 4))
    #         label.set_shape((image_width, image_width, 1))
    #     image = tf.reshape(image, (image_width, image_width, 4))
    #     label = tf.reshape(label, (image_width, image_width, 1))
    #     return image, label

    def preprocess_data(image, label, augmentation_settings):
        if augmentation_settings:
            augmentation_tensor = tf.convert_to_tensor(list(augmentation_settings.values()), dtype=tf.float32)
            image, label = apply_augmentation(image, label, augmentation_tensor)

            image.set_shape((image_width, image_width, 4))
            label.set_shape((image_width, image_width, 1))
        image = tf.reshape(image, (image_width, image_width, 4))
        label = tf.reshape(label, (image_width, image_width, 1))
        return image, label

    def create_dataset(image_paths, label_paths, batch_size, shuffle, augmentation_settings):
        image_tensors = tf.convert_to_tensor(image_paths, dtype=tf.string)
        label_tensors = tf.convert_to_tensor(label_paths, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices((image_tensors, label_tensors))
        dataset = dataset.map(_parse_function,tf.data.experimental.AUTOTUNE) 
        dataset = dataset.map(lambda image, label: preprocess_data(image, label, augmentation_settings),tf.data.experimental.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    cwd = os.path.dirname(__file__)+'/'

    amazon_training_image_dir = cwd+relative_path+'AMAZON/Training/image/'
    amazon_training_label_dir = cwd+relative_path+'AMAZON/Training/label/'

    amazon_validation_image_dir = cwd+relative_path+'AMAZON/Validation/images/'
    amazon_validation_label_dir = cwd+relative_path+'AMAZON/Validation/masks/'

    amazon_test_image_dir = cwd+relative_path+'AMAZON/Test/image/'
    amazon_test_label_dir = cwd+relative_path+'AMAZON/Test/mask/'

    amazon_training_image_paths = [os.path.join(amazon_training_image_dir, filename) for filename in os.listdir(amazon_training_image_dir) if filename.endswith('.tif')]
    amazon_training_label_paths = [os.path.join(amazon_training_label_dir, filename) for filename in os.listdir(amazon_training_label_dir) if filename.endswith('.tif')]

    amazon_validation_image_paths = [os.path.join(amazon_validation_image_dir, filename) for filename in os.listdir(amazon_validation_image_dir) if filename.endswith('.tif')]
    amazon_validation_label_paths = [os.path.join(amazon_validation_label_dir, filename) for filename in os.listdir(amazon_validation_label_dir) if filename.endswith('.tif')]

    amazon_test_image_paths = [os.path.join(amazon_test_image_dir, filename) for filename in os.listdir(amazon_test_image_dir) if filename.endswith('.tif')]
    amazon_test_label_paths = [os.path.join(amazon_test_label_dir, filename) for filename in os.listdir(amazon_test_label_dir) if filename.endswith('.tif')]

    # create the training dataset
    train_dataset = create_dataset(amazon_training_image_paths, amazon_training_label_paths, batch_size=batches, shuffle=shuffle, augmentation_settings=augmentation_settings)
    val_dataset = create_dataset(amazon_validation_image_paths, amazon_validation_label_paths, batch_size=batches, shuffle=shuffle, augmentation_settings=None)
    test_dataset = create_dataset(amazon_test_image_paths, amazon_test_label_paths, batch_size=batches, shuffle=shuffle, augmentation_settings=None)

    return train_dataset, val_dataset, test_dataset, amazon_training_image_paths, amazon_training_label_paths, amazon_validation_image_paths, amazon_validation_label_paths, amazon_test_image_paths, amazon_test_label_paths

# Verify size
# total_objects = 0
# for batch in train_dataset:
#     total_objects += len(batch[0])  # batch[0] contains images, batch[1] contains labels
# print(total_objects)

# Check dimensions of individual images
# first_batch = next(iter(train_dataset.take(1)))
# images, labels = first_batch
# first_image = images[0]
# first_image.shape
# first_image.dtype

# train_dataset.element_spec

# def print_dataset_shapes(dataset, num_samples=100):
#     count = 0
#     for image, label in dataset:
#         print(f"Image shape: {image.shape}, Label shape: {label.shape}")
#         count += 1
#         if count >= num_samples:
#             break

# print("Training dataset shapes:")
# print_dataset_shapes(train_dataset)

# print("Validation dataset shapes:")
# print_dataset_shapes(val_dataset)

# print("Test dataset shapes:")
# print_dataset_shapes(test_dataset)


# import pandas as pd

# pd.DataFrame(columns=['image', 'label',  'batch']).to_csv('../data/diagnose_shape_issue.csv', index=False)
# dt = enumerate(train_dataset)
# for i, (image_batch, label_batch) in dt:
#     batch_size = image_batch.shape[0]
#     try :
#         for j in range(batch_size):
#             image = image_batch[j].numpy()
#             label = label_batch[j].numpy()
#             pd.DataFrame({'image': [image.shape], 'label': [label.shape], 'batch': [i]}).to_csv('../data/diagnose_shape_issue.csv', mode='a', header=False, index=False)
#     except:
#         pd.DataFrame({'image': 'error', 'label': 'error', 'batch': [i]}).to_csv('../data/diagnose_shape_issue.csv', mode='a', header=False, index=False)

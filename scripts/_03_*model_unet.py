import _02c_read_datasets
import _02_evaluate_model
import _02_visualize_predictions

import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras import callbacks
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score

# -------------------- load data

augment = True
train_dataset, val_dataset, test_dataset = _02c_read_datasets.load_datasets(augmented = augment)

# ----------- create directories

out_dir = '../results/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_UNET_AUG/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    os.makedirs(out_dir + '/plots')
    os.makedirs(out_dir + '/weights')
    os.makedirs(out_dir + '/predictions')

checkpoint = callbacks.ModelCheckpoint(
    filepath=out_dir+'weights/'+'model.{epoch:02d}-{val_loss:.4f}.h5',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max', 
    verbose=1
)

# -------------------- define model

def unet(input_shape):
    inputs = tf.keras.Input(input_shape)

    # encoder
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # bridge
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    # decoder
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, (3, 3), padding='same')(up4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, (3, 3), padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(64, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    # output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# -------------------- train model

input_shape = (256, 256, 4)
model = unet(input_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
epochs = 20
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[checkpoint])

# ---------------------- save results

# load best model
model.load_weights(out_dir + 'weights/'+'model.08-0.0480.h5')

# training and validation loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig(out_dir + '/plots/' + 'loss.png')

# training and validation accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.savefig(out_dir + '/plots/' + 'accuracy.png')

# save weights
# model.save(out_dir + '/weights/' + 'model.h5')

# save predictions
def visualize_predictions(index, test_dataset, out_dir, batches = 16):
    
    dir = "image_" + str(index)
    if not os.path.exists(out_dir + '/predictions/' + dir + '/'):
        os.makedirs(out_dir + '/predictions/' + dir + '/')
        os.makedirs(out_dir + '/predictions/' + dir + '/input_image')
        os.makedirs(out_dir + '/predictions/' + dir + '/ground_truth')
        os.makedirs(out_dir + '/predictions/' + dir + '/prediction')
        os.makedirs(out_dir + '/predictions/' + dir + '/prediction_binary')
    
    test_data_iter = iter(itertools.cycle(test_dataset))

    for i in range(index + 1):
        image_batch, label_batch = next(test_data_iter)

    wrapped_index = index % 16
    image = image_batch[wrapped_index].numpy()
    image_rgb = np.stack(
        (
            (image[:,:,0] - np.min(image[:,:,0])) * 255.0 / (np.max(image[:,:,0]) - np.min(image[:,:,0])),
            (image[:,:,1] - np.min(image[:,:,1])) * 255.0 / (np.max(image[:,:,1]) - np.min(image[:,:,1])),
            (image[:,:,2] - np.min(image[:,:,2])) * 255.0 / (np.max(image[:,:,2]) - np.min(image[:,:,2]))
        ),
        axis=-1
    ).astype(np.uint8)

    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    plt.imsave(out_dir + '/predictions/' + dir + '/input_image/' + str(index) + '.png', image_rgb)

    ground_truth = label_batch[wrapped_index].numpy()
    plt.imsave(out_dir + '/predictions/' + dir + '/ground_truth/' + str(index) + '.png', np.squeeze(ground_truth), cmap='gray')

    plt.imsave(out_dir + '/predictions/' + dir + '/prediction/' + str(index) + '.png', np.squeeze(prediction), cmap='gray')

    prediction_binary = np.where(prediction > 0.5, 1, 0)
    plt.imsave(out_dir + '/predictions/' + dir + '/prediction_binary/' + str(index) + '.png', np.squeeze(prediction_binary), cmap='gray')

for i in range(80):
    visualize_predictions(i, test_dataset, out_dir)

# ----------- save metrics

if augment:
    augmetation_settings = {
    "flip_left_right": 0,
    "flip_up_down": 0,
    "gaussian_blur": 0.2,
    "random_noise": 0.0,
    "random_brightness": 0.5,
    "random_contrast": 0.5}
else:
    augmetation_settings = None

batches = 16
shuffled = True

model_info = _02_evaluate_model.evaluate_model(
    "U-net without attention; final dataset", 
    test_dataset,
    model, 
    input_shape, 
    shuffled, 
    batches, 
    epochs, 
    augmentation_settings=augmetation_settings, 
    threshold=0.5
    )
df = pd.DataFrame(model_info)
df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import itertools

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

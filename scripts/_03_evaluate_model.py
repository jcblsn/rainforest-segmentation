from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, jaccard_score
import numpy as np

def evaluate_model(desc_str, test_dataset, model, input_shape, shuffled, batches, epochs, augmentation_settings, threshold=0.5):
    y_true = []
    y_pred = []

    for image_batch, label_batch in test_dataset:
        predictions = model.predict(image_batch)

        for i in range(predictions.shape[0]):
            ground_truth = np.squeeze(label_batch[i].numpy()).flatten()
            prediction = np.squeeze(predictions[i]).flatten()

            y_true.append(ground_truth)
            y_pred.append(prediction > threshold)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    model_info = {'description': [desc_str],
                  'date_saved': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                  'input_shape': [input_shape],
                  'batches': [batches],
                  'epochs': [epochs],
                  'shuffled': [shuffled],
                  'augmentation_settings': [augmentation_settings],
                  'accuracy': [accuracy],
                  'iou': [iou],
                  'f1_score': [f1],
                  'precision': [precision],
                  'recall': [recall]}

    return model_info
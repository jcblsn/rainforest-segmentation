
import tensorflow as tf

def load_datasets(augmented = True):
    if augmented:
        aug = "augmented"
    else:
        aug = "not_augmented"

    train_path = '../data/tf/' + aug + '/train'
    val_path = '../data/tf/' + aug + '/val'
    test_path = '../data/tf/' + aug + '/test'

    train_dataset = tf.data.experimental.load(train_path)
    val_dataset = tf.data.experimental.load(val_path)
    test_dataset = tf.data.experimental.load(test_path)

    return train_dataset, val_dataset, test_dataset


# ex
# train_dataset, val_dataset, test_dataset = load_datasets(augmented = True)
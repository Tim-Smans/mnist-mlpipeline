import tensorflow as tf
import numpy as np
import os

#
# This file will download the MNIST dataset and transform it for training.
##

# Downloading the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize images to range [-1, 1]
train_images = (train_images.astype(np.float32) / 255.0) * 2 - 1
test_images = (test_images.astype(np.float32) / 255.0) * 2 - 1

# Ensure save directory exists
os.makedirs('./data/saves', exist_ok=True)

# Save datasets in TensorFlow-compatible format
np.savez('./data/saves/trainset.npz', images=train_images, labels=train_labels)
np.savez('./data/saves/testset.npz', images=test_images, labels=test_labels)

print("Finished saving data in TensorFlow format")

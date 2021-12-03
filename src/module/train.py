import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model

from .display import DisplayWindow


class Train:
    def __init__(self, parent=None):
        self.display = DisplayWindow(parent)
        self.learning_rate = 0.01
        self.batch_size = 100
        self.checkpoint_path = "data/checkpoint.hdf5"
        self.label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

    def show_img(self):
        plt.figure("Train Images")
        for i in range(9):
            plt.subplot(2, 5, i+1)
            plt.axis("off")
            plt.title(self.label[self.y_train[i][0].astype(int)])
            plt.imshow(self.x_train[i])
        plt.show()

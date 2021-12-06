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

        input_shape = (32, 32, 3)
        self.model = tf.keras.Sequential([
            Conv2D(64, (3, 3), input_shape=input_shape, activation="relu", padding="same"),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation="relu"),
            Dense(4096, activation="relu"),
            Dense(10, activation="softmax")
        ])

    def show_img(self):
        plt.figure("Train Images")
        for i in range(9):
            plt.subplot(2, 5, i+1)
            plt.axis("off")
            plt.title(self.label[self.y_train[i][0].astype(int)])
            plt.imshow(self.x_train[i])
        plt.show()
    
    def show_hyperparameters(self):
        print("hyperparameters:")
        print(f"batch size: {self.batch_size}")
        print(f"learning rate: {self.learning_rate}")
        print("optimizer: SGD")
    
    def show_model(self):
        self.model.summary()

    def show_accuracy(self):
        plt.imshow(plt.imread("data/accu.png"))
        plt.show()
        print("dsa")
    
    def train(self):
        x_train = self.x_train / 255.0
        x_test = self.x_test / 255.0
        y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        y_test = tf.keras.utils.to_categorical(self.y_test, 10)
        # sgd = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=0.0, nesterov=False)
        sgd = optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, decay=0.0, nesterov=False)
        self.model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
        modelCheckpointCallback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=False, monitor="val_accuracy", mode="auto", save_best_only=True)
        hist = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=self.batch_size, callbacks=[modelCheckpointCallback])

        plt.figure("Accuracy and Loss")
        plt.subplot(2, 1, 1)
        plt.plot(hist.history["accuracy"])
        plt.plot(hist.history["val_accuracy"])
        plt.legend(["Training", "Testing"])
        plt.ylabel("accuracy")
        plt.subplot(2, 1, 2)
        plt.plot(hist.history["loss"])
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        plt.savefig("Accuracy and Loss.png")
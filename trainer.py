from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import os

# check that gpu works

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


def create_NN():

    # Build the neural network

    model = Sequential()

    feat_maps = 32
    w, h = 3, 3


    model.add(Conv2D(feat_maps, (h, w), input_shape=(50,50,1), activation="relu"))
    model.add(MaxPooling2D((w,h)))

    model.add(Conv2D(feat_maps, (h, w), activation="relu"))
    model.add(MaxPooling2D((w,h)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # traing the net
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    PATH = "./classes"

    imagegen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.3, validation_split=0.2)

    images = imagegen.flow_from_directory(PATH, class_mode="binary", color_mode="grayscale",
                                          target_size=(50, 50), shuffle=True)

    model.fit(images, epochs=50)


    model.save("catnet.h5")


create_NN()
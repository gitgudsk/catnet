from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
import tensorflow as tf
import os

def create_NN():

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # check that gpu works
    print(device_lib.list_local_devices())
    

    # Build the neural network

    model = Sequential()

    feat_maps = 48
    w, h = 3, 3


    model.add(Conv2D(feat_maps, (h, w), input_shape=(50,50,1), activation="relu"))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(feat_maps, (h, w), activation="relu"))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # traing the net
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    PATH = "./classes"

    imagegen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.2, validation_split=0.2)

    train_gen = imagegen.flow_from_directory(PATH, class_mode="binary", color_mode="grayscale",
                                          target_size=(50, 50), shuffle=True, subset="training")

    valid_gen = imagegen.flow_from_directory(PATH, class_mode="binary", color_mode="grayscale",
                                          target_size=(50, 50), shuffle=True, subset="validation")

    model.fit(train_gen, epochs=40, validation_data=valid_gen)


    model.save("catnet.h5")


create_NN()
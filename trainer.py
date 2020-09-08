from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os

def create_NN():

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    """
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    """

    # check that gpu works
    print(device_lib.list_local_devices())
    

    # Build the neural network

    model = Sequential()

    feat_maps = 32
    w, h = 3, 3


    model.add(Conv2D(feat_maps, (h, w), input_shape=(50,50,3), activation="relu"))
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

    # read the images, training/validation split included
    imagegen = ImageDataGenerator(horizontal_flip=True, zoom_range=0.2, validation_split=0.2)

    train_gen = imagegen.flow_from_directory(PATH, class_mode="binary", color_mode="rgb",
                                          target_size=(50, 50), shuffle=True, subset="training")

    valid_gen = imagegen.flow_from_directory(PATH, class_mode="binary", color_mode="rgb",
                                          target_size=(50, 50), shuffle=True, subset="validation")

    # train the model until it doesn't improve, save the net for later use
    es = EarlyStopping(monitor="val_accuracy", verbose=1, patience=15)
    mc = ModelCheckpoint("catnet.h5", monitor="val_accuracy", save_weights_only=False, save_best_only=True, verbose=1)

    model.fit(train_gen, epochs=100, validation_data=valid_gen, callbacks=[es, mc])


    #model.save("catnet.h5")


create_NN()
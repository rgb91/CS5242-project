"""
Created by Sanjay at 10/31/2018

Feature: Enter feature name here
Enter feature description here
"""
import numpy as np
import os
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, MaxPool2D, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD

TRAIN_DATA_PATH = r'C:\Users\Sanjay Saha\CS5242-project\processed_train_data'
# TRAIN_DATA_PATH = r''
TEST_DATA_PATH = r'C:\Users\Sanjay Saha\CS5242-project\processed_test_data'


def get_model(input_shape=(49, 49, 49, 1), class_num=1):
    """Example CNN

    Keyword Arguments:
        input_shape {tuple} -- shape of input images. Should be (28,28,1) for MNIST and (32,32,3) for CIFAR (default: {(28,28,1)})
        class_num {int} -- number of classes. Shoule be 10 for both MNIST and CIFAR10 (default: {10})

    Returns:
        model -- keras.models.Model() object
    """

    im_input = Input (shape=input_shape)

    t = Convolution3D (32, (24, 24, 24), padding='same') (im_input)  # (24,24,24)
    t = MaxPooling3D (pool_size=(2, 2, 2)) (t)
    t = Convolution3D (64, (12, 12, 12), padding='same') (t)
    t = MaxPooling3D (pool_size=(2, 2, 2)) (t)
    t = Convolution3D (128, (6, 6, 6), padding='same') (t)
    t = Flatten () (t)
    t = Dense (256) (t)
    t = Activation ('relu') (t)
    t = Dense (class_num) (t)
    output = Activation ('softmax') (t)
    model = Model (input=im_input, output=output)
    sgd = SGD (lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile (sgd, 'binary_crossentropy', metrics=['accuracy'])

    return model


def train_data_loader(batch_size=30):
    X = np.load (os.path.join (TRAIN_DATA_PATH, 'X.npy'))
    np.random.shuffle (X)
    y = np.load (os.path.join (TRAIN_DATA_PATH, 'y.npy'))
    np.random.shuffle (y)
    while True:
        start = 0
        end = batch_size
        while end <= X.shape[0]:
            X_mini_batch = X[start:end, :, :, :, :]
            y_mini_batch = y[start:end, :]
            start += batch_size
            end += batch_size
            yield (X_mini_batch, y_mini_batch)


#########################################################################
#                               TRAINING                                #
#########################################################################
X_val = np.load (os.path.join (TRAIN_DATA_PATH, 'X_val.npy'))
y_val = np.load (os.path.join (TRAIN_DATA_PATH, 'y_val.npy'))

n_samples = 200
batch_size = 2

model = get_model (input_shape=(51, 51, 51, 2))
history = model.fit_generator (train_data_loader(batch_size=batch_size), steps_per_epoch=n_samples//batch_size, validation_data=(X_val,  y_val), epochs=1, verbose=1, shuffle=True)


#########################################################################
#                               TESTING                                 #
#########################################################################

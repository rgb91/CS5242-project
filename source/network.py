"""
Created by Sanjay at 10/12/2018

Feature: Neural Network model
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import source.utils as ut
from keras.utils import np_utils
import matplotlib.pyplot as plt

np.random.seed (99)

X_train, y_train, X_test, y_test, class_name = ut.dataset_reader()

X_train = X_train.astype ('float32')
X_test = X_test.astype ('float32')

model = Sequential ()
model.add (Conv2D (32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add (Conv2D (32, (3, 3), activation='relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))
model.add (Dropout (0.25))

model.add (Flatten ())
model.add (Dense (128, activation='relu'))
model.add (Dropout (0.5))
model.add (Dense (10, activation='softmax'))

model.compile (
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit (X_train, y_train, batch_size=32, epochs=1, verbose=1)

score = model.evaluate (X_test, y_test, verbose=0)

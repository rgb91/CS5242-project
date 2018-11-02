"""
Created by Sanjay at 10/12/2018

Feature: Neural Network model
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv3D, AveragePooling3D
import source.utils as ut
from keras.utils import np_utils
import matplotlib.pyplot as plt

np.random.seed (99)

X_train, y_train, X_test, y_test = ut.dataset_reader(datasetpath=r'C:\Users\Sanjay Saha\CS5242-project\processed_data_20')

X_train = X_train.astype ('float32')
X_test = X_test.astype ('float32')

print(X_train.shape)
print(y_train.shape)

model = Sequential ()
model.add (Conv3D (32, (24, 24, 24), activation='relu', input_shape=(49, 49, 49, 2)))
model.add (Conv3D (64, (12, 12, 12), activation='relu'))
model.add (Conv3D (128, (6, 6, 6), activation='relu'))
model.add (AveragePooling3D (pool_size=(2, 2, 2)))
model.add (Dropout (0.25))

model.add (Flatten ())
model.add (Dense (128, activation='relu'))
model.add (Dropout (0.5))
model.add (Flatten ())
model.add (Dense (2, activation='softmax'))

model.compile (
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit_generator (X_train, y_train, batch_size=32, epochs=1, verbose=1, shuffle=False)

score = model.evaluate (X_test, y_test, verbose=0)
print(score)
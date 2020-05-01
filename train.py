"""
"""

import os

# i use this to use amd gpu, comment this line to use defaul tensorflow or something
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Embedding, Reshape
from keras.layers.core import Flatten, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from prepare_data import get_data
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import seed
seed(512)


DATA_LEN = 100

spec, phase, mask = get_data("samples/clean", "samples/noise", mask_size=(128, 109), count=DATA_LEN)

spec -= np.mean(spec)
spec /= np.std(spec)

spec = np.array([spec.T, phase.T]).T

SPEC_SHAPE = spec.shape[1:]
MASK_SHAPE = mask.shape[1]

PADDING = "same"
model = Sequential()

# 1
model.add(Conv2D(16, kernel_size=(3,5), strides=(1,2), activation='relu', input_shape=SPEC_SHAPE))
model.add(Conv2D(32, kernel_size=(3,5), strides=(1,2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,2), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,2), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,2), activation='relu'))


shape = model.output_shape
model.add(Reshape((shape[1], shape[2]*shape[3])))


model.add(Bidirectional(LSTM(512)))


model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense( MASK_SHAPE, activation='sigmoid'))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])


model.fit(spec[:int(DATA_LEN*0.9)], mask[:int(DATA_LEN*0.9)],
                batch_size=4, epochs=5, shuffle=True,
                validation_data=(spec[int(DATA_LEN*0.9):], mask[int(DATA_LEN*0.9):]))

#model.save("m2.h5")
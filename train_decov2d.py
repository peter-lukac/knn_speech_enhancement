"""
"""

import os

# i use this to use amd gpu, comment this line to use defaul tensorflow or something
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Embedding, Reshape, BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from prepare_data import get_data
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import seed
seed(512)


DATA_LEN = 3000

#spec, phase, mask = get_data("samples/clean", "samples/noise", mask_size=(128, 109), count=DATA_LEN)
spec, mask = get_data("samples/clean", "samples/noise", count=DATA_LEN, include_phase=False, flatten=False)

spec.shape = (spec.shape[0], spec.shape[1], spec.shape[2], 1)
mask.shape = (mask.shape[0], mask.shape[1], mask.shape[2], 1)

spec -= np.mean(spec)
spec /= np.std(spec)

#spec = np.array([spec.T, phase.T]).T

SPEC_SHAPE = spec.shape[1:]
MASK_SHAPE = mask.shape[1]

PADDING = "same"
model = Sequential()

ACTIVATION = 'elu'

# 1
model.add(Conv2D(8, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION, input_shape=SPEC_SHAPE))
model.add(Conv2D(16, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))

print(model.output_shape)

model.add(Conv2DTranspose(128, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(64, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(32, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(16, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(1, kernel_size=(3,5), strides=(1,2), activation='hard_sigmoid'))

print(model.output_shape)


model.compile(loss='mse', optimizer='adam', metrics=[])


model.fit(spec[:int(DATA_LEN*0.9)], mask[:int(DATA_LEN*0.9)],
                batch_size=16, epochs=5, shuffle=True,
                validation_data=(spec[int(DATA_LEN*0.9):], mask[int(DATA_LEN*0.9):]))


model.save("decov_3000_elu_hard_sigmoid_5.h5")
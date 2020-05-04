"""
"""

import os

# i use this to use amd gpu, comment this line to use defaul tensorflow or something
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Embedding, Reshape, BatchNormalization, ConvLSTM2D
from keras.layers.core import Flatten, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from prepare_data import get_data
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

from numpy.random import seed
seed(512)


DATA_LEN = 300

SAVE_AS = None
if len(sys.argv) == 2:
    SAVE_AS = sys.argv[1]

#spec, phase, mask = get_data("samples/clean", "samples/noise", mask_size=(128, 109), count=DATA_LEN)
spec, mask = get_data("samples_short/clean", "samples_short/noise",
                count=DATA_LEN, include_phase=False, flatten=False, depth_search=False)

spec.shape = (spec.shape[0], spec.shape[1], spec.shape[2], 1)
mask.shape = (mask.shape[0], mask.shape[1], mask.shape[2], 1)

mean = np.mean(spec)
std = np.std(spec)
spec -= mean
spec /= std


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

shape = model.output_shape

model.add(Reshape((shape[1], shape[2]*shape[3])))
model.add(LSTM(shape[2]*shape[3], return_sequences=True))
model.add(Reshape((shape[1], shape[2], shape[3])))

model.add(Conv2DTranspose(128, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(64, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(32, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(16, kernel_size=(3,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(1, kernel_size=(3,5), strides=(1,2), activation='hard_sigmoid'))



model.compile(loss='mse', optimizer='adam', metrics=[])


h = model.fit(spec[:int(DATA_LEN*0.9)], mask[:int(DATA_LEN*0.9)],
                batch_size=128, epochs=30, shuffle=False,
                validation_data=(spec[int(DATA_LEN*0.9):], mask[int(DATA_LEN*0.9):]))


if SAVE_AS:
    with open(SAVE_AS + ".json", "w") as outfile:
        json.dump(h.history, outfile)

    model.save(SAVE_AS + ".h5")

    with open(SAVE_AS + "_n.json", 'w') as f:
        json.dump({'mean': float(mean), 'std':float(std)}, f)

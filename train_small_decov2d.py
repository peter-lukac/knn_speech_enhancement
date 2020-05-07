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



SAVE_AS = None
#if len(sys.argv) < 3:
#    print("Usage: train_small_decov2d.py CLEAN_DATA_FOLDER NOISE_DATA_FOLDER [MODEL.h5]")
if len(sys.argv) == 4:
    SAVE_AS = sys.argv[3]

#spec, phase, mask = get_data("samples/clean", "samples/noise", mask_size=(128, 109), count=DATA_LEN)
#spec, mask = get_data(sys.argv[1], sys.argv[2],
                #count=DATA_LEN, include_phase=False, flatten=False, depth_search=True)
spec, mask = get_data("data_clean", "data_noise", 11, duration=3*3600, depth_search=True)

DATA_LEN = len(spec)
#spec.shape = (spec.shape[0], spec.shape[1], spec.shape[2], 1)
#mask.shape = (mask.shape[0], mask.shape[1], mask.shape[2], 1)

mean = np.mean(spec)
std = np.std(spec)
spec -= mean
spec /= std


SPEC_SHAPE = spec.shape[1:]
MASK_SHAPE = mask.shape[1]

PADDING = "same"
ACTIVATION = 'elu'

model = Sequential()
# 1
model.add(Conv2D(16, kernel_size=(1,5), strides=(1,3), activation=ACTIVATION, input_shape=SPEC_SHAPE))
model.add(Conv2D(32, kernel_size=(1,5), strides=(1,2), activation=ACTIVATION))
model.add(Conv2D(64, kernel_size=(1,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2D(128, kernel_size=(1,3), strides=(1,2), activation=ACTIVATION))

shape = model.output_shape
print(model.output_shape)

model.add(Reshape((shape[1], shape[2]*shape[3])))
model.add(LSTM(shape[2]*shape[3], return_sequences=True))
model.add(LSTM(shape[2]*shape[3], return_sequences=True))
model.add(Reshape((shape[1], shape[2], shape[3])))

model.add(Conv2DTranspose(128, kernel_size=(1,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(64, kernel_size=(1,5), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(32, kernel_size=(1,5), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(1, kernel_size=(1,5), strides=(1,3), activation='hard_sigmoid'))
print(model.output_shape)


model.compile(loss='mse', optimizer='adam', metrics=[])
model.save_weights('tmp/w.h5')

h = model.fit(spec[:int(DATA_LEN*0.9)], mask[:int(DATA_LEN*0.9)],
                batch_size=32, epochs=5, shuffle=False,
                validation_data=(spec[int(DATA_LEN*0.9):], mask[int(DATA_LEN*0.9):]))


if SAVE_AS:
    if ".h5" in SAVE_AS:
        SAVE_AS = SAVE_AS.split(".h5")[0]
    with open(SAVE_AS + ".json", "w") as outfile:
        json.dump(h.history, outfile)
    model.save(SAVE_AS + ".h5")
    with open(SAVE_AS + "_n.json", 'w') as f:
        json.dump({'mean': float(mean), 'std':float(std)}, f)


"""
"""

import os

# I use this to use amd gpu, comment this line to use defaul tensorflow or something
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Embedding, Reshape
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
if len(sys.argv) == 5:
    folder_clean = sys.argv[1]
    folder_noise = sys.argv[2]
    SAVE_AS = sys.argv[3]
    epochs = int(sys.argv[4])
elif len(sys.argv) == 1:
    print("Using default clean folder(\"data_clean\") and default noise folder(\"data_noise\")")
    folder_clean = "data_clean"
    folder_clean = "data_noise"
    epochs = 1
else:
    print("Usage: ...")
    sys.exit(1)


spec, mask = get_data(folder_clean,folder_clean, 27, 8*3600, depth_search=False)

DATA_LEN = len(spec)

mean = np.mean(spec)
std = np.std(spec)
spec -= mean
spec /= std

SPEC_SHAPE = spec.shape[1:]

ACTIVATION = 'elu'

model = Sequential()

model.add(Conv2D(16, kernel_size=(1,5), strides=(1,3), activation=ACTIVATION, input_shape=SPEC_SHAPE))
model.add(Conv2D(32, kernel_size=(1,5), strides=(1,2), activation=ACTIVATION))
model.add(Conv2D(64, kernel_size=(1,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2D(128, kernel_size=(1,3), strides=(1,2), activation=ACTIVATION))

shape = model.output_shape

model.add(Reshape((shape[1], shape[2]*shape[3])))
model.add(LSTM(shape[2]*shape[3], return_sequences=True))
model.add(LSTM(shape[2]*shape[3], return_sequences=True))
model.add(Reshape((shape[1], shape[2], shape[3])))

model.add(Conv2DTranspose(128, kernel_size=(1,3), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(64, kernel_size=(1,5), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(32, kernel_size=(1,5), strides=(1,2), activation=ACTIVATION))
model.add(Conv2DTranspose(1, kernel_size=(1,5), strides=(1,3), activation='hard_sigmoid'))

model.compile(loss='mse', optimizer='adam', metrics=[])


h = model.fit(spec[:int(DATA_LEN*0.9)], mask[:int(DATA_LEN*0.9)],
                batch_size=64, epochs=epochs,
                validation_data=(spec[int(DATA_LEN*0.9):], mask[int(DATA_LEN*0.9):]))


if SAVE_AS:
    if ".h5" in SAVE_AS:
        SAVE_AS = SAVE_AS.split(".h5")[0]
    with open(SAVE_AS + ".json", "w") as outfile:
        json.dump(h.history, outfile)
    model.save(SAVE_AS + ".h5")
    with open(SAVE_AS + "_n.json", 'w') as f:
        json.dump({'mean': float(mean), 'std':float(std)}, f)

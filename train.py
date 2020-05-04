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
import json
import sys

from numpy.random import seed
seed(512)


DATA_LEN = 12000

SAVE_AS = None
if len(sys.argv) == 2:
    SAVE_AS = sys.argv[1]


#spec, phase, mask = get_data("samples/clean", "samples/noise", mask_size=(128, 109), count=DATA_LEN)
spec, mask = get_data("samples/clean", "samples/noise", mask_size=(128, 109), count=DATA_LEN, include_phase=False)
spec.shape = (spec.shape[0], spec.shape[1], spec.shape[2], 1)

mean = np.mean(spec)
std = np.std(spec)
spec -= mean
spec /= std

#spec = np.array([spec.T, phase.T]).T

SPEC_SHAPE = spec.shape[1:]
MASK_SHAPE = mask.shape[1]

PADDING = "same"
model = Sequential()

# 1
model.add(Conv2D(16, kernel_size=(3,5), strides=(1,2), activation='relu', input_shape=SPEC_SHAPE))
model.add(Conv2D(32, kernel_size=(3,5), strides=(1,2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,2), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,2), activation='relu'))
model.add(Conv2D(8, kernel_size=(3,3), strides=(1,2), activation='relu'))


model.add(Flatten())
# removed LSTM due to extremly long training time and bad results

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense( MASK_SHAPE, activation='sigmoid'))

model.compile(loss='mse', optimizer='adam', metrics=[])


model.fit(spec[:int(DATA_LEN*0.9)], mask[:int(DATA_LEN*0.9)],
                batch_size=16, epochs=10,
                validation_data=(spec[int(DATA_LEN*0.9):], mask[int(DATA_LEN*0.9):]))



if SAVE_AS:
    with open(SAVE_AS + ".json", "w") as outfile:
        json.dump(h.history, outfile)

    model.save(SAVE_AS + ".h5")

    with open(SAVE_AS + "_n.json", 'w') as f:
        json.dump({'mean': float(mean), 'std':float(std)}, f)

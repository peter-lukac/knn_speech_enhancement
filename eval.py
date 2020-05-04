import librosa
import soundfile as sf
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
import sys
import json
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import load_model

SPEC_MIN = 1e-13
SPEC_MIN_LOG = 13

model_name = "models/short_2000_elu_hard_sigmoid_81.h5"

model = load_model(model_name)

with open(model_name.split('.')[0] + "_n.json", 'r') as f:
    norm = json.load(f)
mean = norm["mean"]
std = norm["std"]

if len(sys.argv) != 4:
    print("Usage: ...")
    sys.exit(1)

clean_folder = sys.argv[1]
noise_folder = sys.argv[2]
synth_folder = sys.argv[3]

if clean_folder[-1] != "/":
    clean_folder += "/"

if noise_folder[-1] != "/":
    noise_folder += "/"

if synth_folder[-1] != "/":
    synth_folder += "/"

clean_files = os.listdir(clean_folder)
noise_files = os.listdir(noise_folder)


for clean in clean_files:
    for noise in noise_files:
        x1, fs = sf.read(clean_folder + clean)
        x2, fs = sf.read(noise_folder + noise)
        if len(x2) > len(x1):
            x2 = x2[:len(x1)]
        if len(x2) < len(x1):
            z = np.zeros((x1.shape))
            z[:len(x2)] =  x2
            x2 = z
        name = synth_folder + clean + "_" + noise
        try:
            os.mkdir(name)
        except FileExistsError:
            pass
        x = x1 + x2
        sf.write(name + "/" + "noisy.flac", x, fs)
        sf.write(name + "/" + "target.flac", x1, fs)

        spec = np.abs(librosa.stft(x, n_fft=512, hop_length=250, win_length=512))**2
        spec = np.maximum(spec, SPEC_MIN)
        spec = np.log10(spec) + SPEC_MIN_LOG
        spec = spec.T
        length = model.input_shape[1]
        if len(spec)/length != int(len(spec)/length):
            padded_length = int((len(spec)/length) + 1) * length
            pad = np.zeros((padded_length-len(spec), spec.shape[1]))
            spec = np.concatenate((spec, pad))
        spec = spec.reshape(int(len(spec)/length), length, spec.shape[1], 1)
        spec -= mean
        spec /= std
        mask = model.predict(spec)
        mask = np.concatenate(mask)
        spec = np.concatenate(spec)

        clean_spec = mask*spec
        clean_spec.shape = (clean_spec.shape[0], clean_spec.shape[1])
        clean_spec = clean_spec.T
        clean_spec *= std
        clean_spec += mean
        clean_spec -= SPEC_MIN_LOG
        clean_spec = np.power(10, clean_spec)
        mel = librosa.feature.melspectrogram(sr=fs,  n_fft=512, S=clean_spec)
        audio = librosa.feature.inverse.mel_to_audio(mel, sr=fs, n_fft=512, hop_length=250)
        sf.write(name + "/" + "synth.flac", audio, fs)
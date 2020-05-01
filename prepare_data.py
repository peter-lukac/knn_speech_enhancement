import librosa
import soundfile as sf
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2


SPEC_MIN = 1e-13
SPEC_MIN_LOG = 13


CLEAN_FOLDER = "test/clean"
NOISE_FOLDER = "test/noise"


def yield_noise(noise_folder):
    noise_folders = os.walk(noise_folder)
    noise_folders = list(filter(lambda x: not x[2] == [], noise_folders))
    file_counter = {}
    for f in noise_folders:
        file_counter[f[0]] = 0
    while True:
        for (folder, fos, f) in noise_folders:
            yield folder + "/" + f[file_counter[folder]]
            file_counter[folder] += 1
            if file_counter[folder] >= len(f):
                file_counter[folder] = 0



def get_data(clean_folder, noise_folder, start=0, count=np.inf, mask_size=None, include_phase=True, flatten=True):
    idx = 0
    noise_array = []
    phase_array = []
    mask_array = []
    for f_clean, f_noise in zip(os.listdir(clean_folder), yield_noise(noise_folder)):
        if idx < start:
            idx += 1
            continue
        #print(f_clean + "\t" + f_noise)
        x_clean, fs_clean = sf.read(clean_folder + "/" + f_clean)
        x_noise, fs_noise = sf.read(f_noise)
        x = x_clean + x_noise
        spec_clean = np.abs(librosa.stft(x_clean, n_fft=512, hop_length=250, win_length=512))**2
        stft_noise = librosa.stft(x, n_fft=512, hop_length=250, win_length=512)
        spec_noise = np.abs(stft_noise)**2
        phase = np.angle(stft_noise)
        #m = np.nanmin(spec[spec != 0])
        spec_clean = np.maximum(spec_clean, SPEC_MIN)
        spec_noise = np.maximum(spec_noise, SPEC_MIN)

        spec_clean = np.log10(spec_clean) + SPEC_MIN_LOG
        spec_noise = np.log10(spec_noise) + SPEC_MIN_LOG
        spec_noise = np.maximum(spec_noise, SPEC_MIN)
        mask = spec_clean / spec_noise

        mask = np.minimum(mask, 1)
        noise_array.append(spec_noise.T)

        if include_phase:
            phase_array.append(phase.T)

        if mask_size:
            mask = cv2.resize(mask, dsize=(mask_size[1], mask_size[0]))

        if flatten:
            mask_array.append(mask.T.flatten())
        else:
            mask_array.append(mask.T)

        idx += 1
        if idx >= start + count:
            break

    if include_phase:
        return np.array(noise_array), np.array(phase_array), np.array(mask_array)
    else:
        return np.array(noise_array), np.array(mask_array)


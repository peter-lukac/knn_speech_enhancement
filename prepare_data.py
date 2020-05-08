import librosa
import soundfile as sf
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage.transform import resize


SPEC_MIN = 1e-13
SPEC_MIN_LOG = 13



def yield_noise(noise_folder, depth_search=False):
    noise_folders = os.walk(noise_folder)
    noise_folders = list(filter(lambda x: not x[2] == [], noise_folders))
    if depth_search:
        while True:
            for (folder, fos, files) in noise_folders:
                for f in files:
                    yield folder + "/" + f
    file_counter = {}
    for f in noise_folders:
        file_counter[f[0]] = 0
    while True:
        for (folder, fos, f) in noise_folders:
            yield folder + "/" + f[file_counter[folder]]
            file_counter[folder] += 1
            if file_counter[folder] >= len(f):
                file_counter[folder] = 0



def get_data_old(clean_folder, noise_folder, start=0, count=np.inf,
                mask_size=None, include_phase=True, flatten=True, depth_search=False):
    idx = 0
    noise_array = []
    phase_array = []
    mask_array = []
    for f_clean, f_noise in zip(os.listdir(clean_folder), yield_noise(noise_folder, depth_search)):
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
            mask = resize(mask, (mask_size[1], mask_size[0]))

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


class AudioFetch:
    def __init__(self, noise_folder):
        og_folders = os.walk(noise_folder)
        og_folders = list(filter(lambda x: any(map(lambda y: ".flac" in y or ".wav" in y, x[2])), og_folders))
        self.folder_counter = 0
        self.folders = []

        if og_folders == []:
            print("Empty audio folders")
            sys.exit(2)

        for f in og_folders:
            self.folders.append([f[0], list(filter(lambda x: ".flac" in x or ".wav" in x, f[2])), 0])

    def get(self, depth_search=False):
        i = self.folder_counter
        f = self.folders[i][0] + "/" + self.folders[i][1][self.folders[i][2]]
        self.folders[i][2] += 1
        if depth_search is False:
            self.folder_counter += 1
        if self.folders[i][2] >= len(self.folders[i][1]):
            self.folders[i][2] = 0
            if depth_search is True:
                self.folder_counter += 1
        if self.folder_counter >= len(self.folders):
                self.folder_counter = 0
        return f


def get_data_of_duration(folder, depth_search, duration, target_fs=16000):
    d = 0
    data = []
    fetch = AudioFetch(folder)
    while d < duration:
        x, fs = sf.read(fetch.get(depth_search))
        d += len(x)/fs
        if len(x.shape) == 2:
            x = np.mean(x, axis=1)
        if fs != target_fs:
            x = librosa.resample(x, fs, target_fs)
        data.append(x)
    return np.concatenate(data)


def get_data(clean_folder, noise_folder, size, duration=3600, fs=16000,
                depth_search=False, dense=False):
    x_clean = get_data_of_duration(clean_folder, True, duration, fs)
    x_noise = get_data_of_duration(noise_folder, depth_search, duration, fs)

    if len(x_clean) <= len(x_noise):
        x_noise = x_clean + x_noise[:len(x_clean)]
    else:
        x_noise = x_clean[:len(x_noise)] + x_noise
        x_clean = x_clean[:len(x_noise)]


    spec_clean = np.abs(librosa.stft(x_clean, n_fft=512, hop_length=250, win_length=512))**2
    spec_noise = np.abs(librosa.stft(x_noise, n_fft=512, hop_length=250, win_length=512))**2

    spec_clean = np.maximum(spec_clean, SPEC_MIN)
    spec_noise = np.maximum(spec_noise, SPEC_MIN)
    spec_clean = np.log10(spec_clean) + SPEC_MIN_LOG
    spec_noise = np.log10(spec_noise) + SPEC_MIN_LOG
    spec_noise = np.maximum(spec_noise, SPEC_MIN)

    mask = spec_clean / spec_noise
    mask = np.minimum(mask, 1)

    spec_noise = spec_noise.T
    mask = mask.T

    length = int(len(spec_noise) / size)
    spec_noise = spec_noise[0:size*length]
    mask = mask[0:size*length]
    spec_noise = spec_noise.reshape((length, size, 257, 1))
    mask = mask.reshape((length, size, 257, 1))

    return spec_noise, mask

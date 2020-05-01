import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os


def ok_path(path, skip_folders=[]):
    for folder in skip_folders:
        if folder in path:
            return False
    return True


def is_sound(filename):
    for i in ['.wav', '.flac']:
        if i in filename:
            return True
    return False


def prepare_samples(src_folder, tgt_folder,
                    length=3, min_length=1,
                    keep_first=True, tgt_sample_freq=16000,
                    skip_folders=[], level=1.0):
    if not tgt_folder[-1] == "/":
        tgt_folder = tgt_folder + "/"
    for (dirpath, dirnames, filenames) in os.walk(src_folder):
        if not ok_path(dirpath, skip_folders):
            continue
        filenames = filter(lambda x: is_sound(x), filenames)
        for f in filenames:
            file_counter = 0
            x, fs = sf.read(dirpath + "\\" + f)
            if len(x.shape) == 2:
                x = np.mean(x, axis=1)
            if len(x) < 8000:
                continue
            x *= level
            if not fs == tgt_sample_freq:
                x = librosa.core.resample(x, fs, tgt_sample_freq)
            step = tgt_sample_freq*length
            for i1, i2 in zip(np.arange(0, len(x)+1, step), np.arange(step, len(x)+step, step)):
                i2 = min(len(x), i2)
                if (i2-i1)/tgt_sample_freq < min_length:
                    if not i1 == 0 or keep_first is False:
                        break
                if (i2-i1)/tgt_sample_freq < length:
                    x2 = np.zeros((length*tgt_sample_freq))
                    x2[0:i2-i1] = x[i1:i2]
                else:
                    x2 = x[i1:i2]
                #librosa.output.write_wav(tgt_folder + f + "_" + str(file_counter) + ".wav", x2, tgt_sample_freq)
                sf.write(tgt_folder + f + "_" + str(file_counter) + ".flac", x2, tgt_sample_freq)
                file_counter += 1

           
prepare_samples("LibriSpeech/dev-clean", "samples/clean")
prepare_samples("LibriSpeech/dev-other", "samples/clean")

prepare_samples("Nonspeech", "samples/noise/random", level=0.2)

prepare_samples("qutnoisecafe", "samples/noise/cafe", min_length=3, keep_first=False)
prepare_samples("qutnoisecar", "samples/noise/car", min_length=3, keep_first=False)
prepare_samples("qutnoisehome", "samples/noise/home", min_length=3, keep_first=False)
prepare_samples("qutnoisereverb", "samples/noise/reverb", min_length=3, keep_first=False)
prepare_samples("qutnoisestreet", "samples/noise/street", min_length=3, keep_first=False)

prepare_samples("BabbleNoise", "samples/noise/babble", min_length=3, keep_first=False)
prepare_samples("DrivingcarNoise", "samples/noise/driving", min_length=3, keep_first=False)
prepare_samples("MachineryNoise", "samples/noise/machines", min_length=3, keep_first=False)
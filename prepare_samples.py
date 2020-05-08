import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path


def is_sound(filename):
    return '.wav' in filename or '.flac' in filename


def prepare_samples(src_folder, tgt_folder, length=3, min_length=1,
                    keep_first=True, tgt_sample_freq=16000, level=1.0):
    if not tgt_folder[-1] == "/":
        tgt_folder = tgt_folder + "/"
    try:
        os.makedirs(tgt_folder)
    except FileExistsError:
        pass
    for (dirpath, dirnames, filenames) in os.walk(src_folder):
        filenames = filter(is_sound, filenames)
        for f in filenames:
            file_counter = 0
            x, fs = sf.read(os.path.join(dirpath, f))
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
                sf.write(os.path.join(tgt_folder, f + "_" + str(file_counter) + ".flac"), x2, tgt_sample_freq)
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



prepare_samples("LibriSpeech/dev-clean", "samples_short/clean", length=0.32, min_length=0.32)
prepare_samples("LibriSpeech/dev-other", "samples_short/clean", length=0.32, min_length=0.32)

prepare_samples("Nonspeech", "samples_short/noise/random", level=0.2, length=0.32, min_length=0.32)

prepare_samples("qutnoisecafe", "samples_short/noise/cafe", length=0.32, min_length=0.32)
prepare_samples("qutnoisecar", "samples_short/noise/car", length=0.32, min_length=0.32)
prepare_samples("qutnoisehome", "samples_short/noise/home", length=0.32, min_length=0.32)
prepare_samples("qutnoisereverb", "samples_short/noise/reverb", length=0.32, min_length=0.32)
prepare_samples("qutnoisestreet", "samples_short/noise/street", length=0.32, min_length=0.32)

prepare_samples("BabbleNoise", "samples_short/noise/babble", length=0.32, min_length=0.32)
prepare_samples("DrivingcarNoise", "samples_short/noise/driving", length=0.32, min_length=0.32)
prepare_samples("MachineryNoise", "samples_short/noise/machines", length=0.32, min_length=0.32)

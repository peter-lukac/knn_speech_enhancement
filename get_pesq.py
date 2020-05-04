import soundfile as sf
import numpy as np
import os
import sys
from pesq import pesq


if len(sys.argv) != 2:
    print("Usage: get_pesq.py FOLDER")
    sys.exit(1)


folders = os.walk(sys.argv[1])
folders = list(filter(lambda x: not x[2] == [], folders))

for folder, ignored, files in folders:
    ref, fs = sf.read(folder + "/" + files[files.index('target.flac')])
    noise, fs = sf.read(folder + "/" + files[files.index('noisy.flac')])
    synth, fs = sf.read(folder + "/" + files[files.index('synth.flac')])

    ref_nb = pesq(fs, ref, ref, 'nb')
    ref_wb = pesq(fs, ref, ref, 'wb')

    noise_nb = pesq(fs, ref, noise, 'nb')
    noise_wb = pesq(fs, ref, noise, 'wb')

    synth_nb = pesq(fs, ref, synth, 'nb')
    synth_wb = pesq(fs, ref, synth, 'wb')
    print(folder)
    print("PESQ wide band:   ref:" + str(ref_wb) + "\tnoisy:" + str(noise_wb) + "\tsynthesized:" + str(synth_wb))
    print("PESQ narrow band: ref:" + str(ref_nb) + "\tnoisy:" + str(noise_nb) + "\tsynthesized:" + str(synth_nb))
    print()
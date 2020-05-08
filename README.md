# Speech enhancement using convolutional neural networks

## Instructions

### Installation
If your system already runs `conda` you can skip the first two steps, otherwise run
the rest inside the `miniconda3` container.

```bash
docker run -t -i continuumio/miniconda3 /bin/bash
apt-get install unzip
conda install -y -c conda-forge tensorflow keras numpy scikit-image matplotlib libsndfile alsa-lib libsndfile
conda install -y -c auto gl
conda install -y portaudio
pip install plaidml-keras plaidbench librosa soundfile sounddevice
plaidml-setup
```

### Fetch and prepare dataset
Fetch the LibriSpeech corpus, as well as a number of standard noise datasets,
unpack them, and prepare them for training and evaluation.

```bash
curl -q http://www.openslr.org/resources/12/dev-clean.tar.gz | tar xvz
curl -q http://www.openslr.org/resources/12/dev-other.tar.gz | tar xvz
curl -q http://www.openslr.org/resources/12/test-clean.tar.gz | tar xvz
curl -q http://www.openslr.org/resources/12/test-other.tar.gz | tar xvz

wget https://personal.utdallas.edu/~nxk019000/VAD-dataset/BabbleNoise.zip
wget https://personal.utdallas.edu/~nxk019000/VAD-dataset/DrivingcarNoise.zip
wget https://personal.utdallas.edu/~nxk019000/VAD-dataset/MachineryNoise.zip
wget http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/Nonspeech.zip
wget https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/9b0f10ed-e3f5-40e7-b503-73c2943abfb1/download/qutnoisecafe.zip
wget https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/7412452a-92e9-4612-9d9a-6b00f167dc15/download/qutnoisecar.zip
wget https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/35cd737a-e6ad-4173-9aee-a1768e864532/download/qutnoisehome.zip
wget https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/164d38a5-c08e-4e20-8272-793534eb10c7/download/qutnoisereverb.zip
wget https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/10eeceae-9f0c-4556-b33a-dcf35c4f4db9/download/qutnoisestreet.zip
find . -name '*.zip' -exec unzip {} \; && rm *.zip

python prepare_samples.py
```

### Train
Train 64 epochs and save the result to model.h5.

```bash
python train.py samples/clean samples/noise model.h5 64
```

### Evaluate and synthesize
Synthesize the cleaned samples into the folder `synth`.

```bash
mkdir synth
python eval.py model.h5 samples/clean samples/noise synth
```

### Evaluate PESQ values
```bash
python get_pesq.py synth
```

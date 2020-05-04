# knn_speech_enhancement


1. Download data:

Clean speech data. Extract them in the root repository folder
http://www.openslr.org/resources/12/dev-clean.tar.gz
http://www.openslr.org/resources/12/dev-other.tar.gz
http://www.openslr.org/resources/12/test-clean.tar.gz
http://www.openslr.org/resources/12/test-other.tar.gz
 

Noise data. Extract them in the root repository folder
https://personal.utdallas.edu/~nxk019000/VAD-dataset/BabbleNoise.zip
https://personal.utdallas.edu/~nxk019000/VAD-dataset/DrivingcarNoise.zip
https://personal.utdallas.edu/~nxk019000/VAD-dataset/MachineryNoise.zip

http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/Nonspeech.zip

https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/9b0f10ed-e3f5-40e7-b503-73c2943abfb1/download/qutnoisecafe.zip
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/7412452a-92e9-4612-9d9a-6b00f167dc15/download/qutnoisecar.zip
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/35cd737a-e6ad-4173-9aee-a1768e864532/download/qutnoisehome.zip
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/164d38a5-c08e-4e20-8272-793534eb10c7/download/qutnoisereverb.zip
https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/10eeceae-9f0c-4556-b33a-dcf35c4f4db9/download/qutnoisestreet.zip


2. Run prepare_samples.py

3. Train:
    model with 2d convolutions and FC layers: python train.py MODEL.h5
    model with 2d convolutions and 2d deconvolutions: python train_decov2d.py MODEL.h5
    model with 2d convolutions and 2d deconvolutions for smaller input: python train_small_decov2d.py MODEL.h5

4. Create eval data and  synthesize results:
    python eval.py MODEL.h5 CLEAN_FOLDER NOISE_FOLDER SYNTH_TARGET_FOLDER
    MODEL.h5 trained model for speech enhancing
    CLEAN_FOLDER folder with clean samples
    NOISE_FOLDER with noise samples
    SYNTH_TARGET_FOLDER target folder that will contain synthesized samples
    E.g. python eval.py model.h5 eval/clean eval/noise synth

5. Get PESQ values:
    E.g. python get_pesq.py synth

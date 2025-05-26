# MusicGen Music Continuation  
## Installation 
### Create Conda Environment 
```
conda create -n MusicGen python=3.9
```
### Install Packages  
```
python -m pip install 'torch==2.1.0'
python -m pip install setuptools wheel
python -m pip install -U audiocraft
```
## Inference  
under `MusicGen-Continuation` folder, run:
```
python inference.py --reference_audio <path to reference audio> --reference_duration <desired audio length of reference> (...other arguments)
```
for example:  
```
python inference.py --use_sampling --progress --reference_audio ./example/bach.mp3 --reference_duration 10.0 --text_prompt "jazz style" --cfg_coef 5.0
```
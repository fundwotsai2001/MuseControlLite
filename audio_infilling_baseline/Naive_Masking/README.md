# Naive Masking  
## Installation 
### Create Conda Environment 
```
conda create -n Naive-Masking python=3.9
```
### Install Packages  
```
python -m pip install diffusers transformers torch torchaudio torchsde
python -m pip install -U "huggingface_hub[cli]"
```
### Huggingface Login
```
huggingface-cli login
```
login using your Access Token
## Inference  
under `Naive-Masking` folder, run:
```
python inference.py --reference_audio <path to reference audio> --text_prompt <text prompt> (...other arguments)
```
for example:  
```
python inference.py --reference_audio ./example/bach.mp3 --text_prompt "jazz style" --cfg_coef 7.0
```
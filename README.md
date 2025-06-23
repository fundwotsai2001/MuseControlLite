# MuseControlLite

This is the official implementation of MuseControlLite (ICML2025).
[paper]() | [demo](https://musecontrollite.github.io/web/) | [colab](https://colab.research.google.com/drive/1rR-Ncng_gSeb6hX0LY20SA4O9RCF-ZF3)
![Alt Text](Melody_result.png)
We have updated the results for MuseControlLite-stereo-Melody, which processes the melody of the two audio channels separately and delivers an additional performance boost. By contrast, the results reported in the paper mixed the melody from both channels.

## Todo (updated 6/23)
- [x] [Colab impementation](https://colab.research.google.com/drive/1rR-Ncng_gSeb6hX0LY20SA4O9RCF-ZF3?usp=sharing) (Thanks to [YianLai0327](https://github.com/YianLai0327))
- [x] MuseControlLite with the diffusers libary
- [ ] MuseControlLite with the stable-audio-tools libary
- [ ] Put model on Huggingface
- [ ] Provide scale-up version

## Installation
We provide a step by step series of examples that tell you how to get a development environment running.
```
git clone https://github.com/fundwotsai2001/MuseControlLite.git
cd MuseControlLite

## Install environment
conda create -n MuseControlLite python=3.11
conda activate MuseControlLite
pip install -r requirements.txt
sudo apt install ffmpeg # For Linux

## Install checkpoint
gdown 1Q9B333jcq1czA11JKTbM-DHANJ8YqGbP --folder
```
## huggingface-cli login
To use the stable-audio open 1.0 model, you will need to a token generated from [higgingface](https://huggingface.co/settings/tokens).
```
huggingface-cli login
```
## Inference
All the hyper-parameters could be found in `config_inference.py`, we provide detailed comments as guidance, especially, you can use set the conditions for `"condition_type"`. Run:
```
python MuseControlLite_inference.py
```


## Finetuning with your own dataset
### Caption labeling
We use [Jamendo](https://github.com/MTG/mtg-jamendo-dataset), we provide the code for [Text-to-music-dataset-preparation](https://github.com/fundwotsai2001/Text-to-music-dataset-preparation).
You can use the pipeline for other audio datasets as well. 
### Condition extraction
[Text-to-music-dataset-preparation](https://github.com/fundwotsai2001/Text-to-music-dataset-preparation) will output a json file that records text-audio pair with a list of dictionaries as below:
```
[
    {
        "path": "88/1394788_chunk0.mp3",
        "Qwen_caption": "The music piece is an instrumental blend of folk and psychedelic rock genres, set in A minor with a tempo of around 105 BPM. It features a 4/4 time signature and a complex chord progression. The mood evoked is one of tension and unease, perhaps suitable for a suspenseful scene in a film or play. The instruments include guitar, bass, drums, and a flute, contributing to the overall texture and atmosphere of the track."
    },
    .
    .
    .
]
```
Set `--audio_folder`, `--meta_path`, `--new_json`, and run:
```
python extract_musical_attribute_conditions.py --audio_folder "../mtg_full_47s" --meta_path "./Qwen_caption.json" "--new_json" "./test_condition.json"
python extract_musical_attribute_conditions_melody_stereo.py --audio_folder "../mtg_full_47s" --meta_path "./test_condition.json" "--new_json" "./test_condition_stereo_melody.json"
```
This will extract the conditions so that you don't have to do it on the fly during training.
### VAE extraction
The training code used preprocessed latents, so that we can skip the VAE encoding during training. To preprocess latents, run:
```
python stable_audio_VAE_encode.py --audio_folder "../mtg_full_47s" --meta_path "./Qwen_caption.json" --latent_dir "./Jamendo_audio_47s_latent" --batch_size 1
```
### MuseControlLite training
*All training hyper-parameters could be found in `config_training.py`. 
If you want to train a model that can deal with all musical attribute conditions, simply set `"condition_type": ["dynamics", "rhythm", "melody"]`, and run:
```
python MuseControlLite_train_all.py
# You can try different combinations for the 'condition_type', the conditions that are not selected will be filled with zero as unconditioned. 
# We found that training together with the audio condition produces unsatisfactory results.
```

If you only want to train on one specific condition, you might try this:
```
python MuseControlLite_train_melody_stereo.py
# You can modify the code for other conditions.
```
For audio in-painting and out-painting:
```
python MuseControlLite_train_audio_only.py
# This set of adapters can then cooperate with the adapters trained with musical attribute conditions.
```
In our experiment, usually, if using all the conditions that are used in training during inference, the FAD will be lower. Thus if you only need a certain conditions, then it is recommended just train that condition.
# MuseControlLite

This is the official implementation of MuseControlLite.

## Installation
We provide a step by step series of examples that tell you how to get a development environment running.
```
git clone https://github.com/fundwotsai2001/MuseControlLite_v2.git
cd MuseControlLite_v2

## Install environment
conda create -n stable-audio python=3.11
conda activate stable-audio
pip install -r requirements.txt

## Install checkpoint
gdown 1Q9B333jcq1czA11JKTbM-DHANJ8YqGbP --folder
```

## Inference
All the hyper-parameters could be found in `config_inference.py`, we provide detailed comments as guidance. Run:
```
MuseControlLite_inference_on_the_fly_all_together.py # Capable for all conditions
```
If you only need melody condition, simply set `"condition_type": ["melody"]` and run:
```
python MuseControlLite_inference_on_the_fly_melody.py # Specialized on melody
```
## Finetuning with your own dataset
### Caption labeling
We use [Jamendo](https://github.com/MTG/mtg-jamendo-dataset), we will provide the code pipeline in the future. You can:
1. Download the full-length version from [Jamendo](https://github.com/MTG/mtg-jamendo-dataset)
2. Resample the audio to 44100 hz, and slice it to shape (2, 2097152)
3. Use sound event detection in Panns to filter out audio that contains vocal.
4. Use `Qwen/Qwen2-Audio-7B-Instruct` to obtain all captions for the audio.
5. Filter out audios that appear in the no vocal Song Describer Dataset.


You can use the pipeline for other audio datasets as well. The code for the caption labeling will be provided [here]().
### Condition extraction
Prepare text-audio pair with a list of dictionaries as below:
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
```
This will extract the conditions so that you don't have to do it on the fly during training.
### VAE extraction
Set `--audio_folder`, `--meta_path`, `--latent_dir`, `--batch_size`, and run:
```
python stable_audio_VAE_encode.py --audio_folder "../mtg_full_47s" --meta_path "./Qwen_caption.json" --latent_dir "./Jamendo_audio_47s_latent" --batch_size 1
```
### MuseControlLite training
*All training settings could be found in `config_training.py`. 
If you want to train a model that can deal with all functions, simply set `"condition_type": ["dynamics", "rhythm", "melody", "audio"]`, and run:
```
python MuseControlLite_train_all.py
# You can try different combinations for the 'condition_type', the conditions that are not selected will be filled with zero as unconditioned. 
```

If you only want to train on one condition, you might try this:
```
python MuseControlLite_train_melody_only.py
# You can modify the code for to other conditions.
```
For audio in-painting and out-painting:
```
python MuseControlLite_train_audio_only.py
```
In our experiment, usually, if using all the conditions that are used in training during inference, the FAD will be lower. Thus if you only need a certain conditions, then it is recommended just train that condition.
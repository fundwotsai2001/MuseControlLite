import os
from config_inference_infilling import get_config
from tqdm import tqdm
config = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torchaudio import transforms as T
from utils.stable_audio_dataset_utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
import json
import torch.nn.functional as F
class AudioInversionDataset(Dataset):
    def __init__(
        self,
        config,
        audio_data_root,
        device,
        force_channels="stereo"
    ):
        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )
        self.root_paths = []
        self.force_channels = force_channels
        self.config = config
        self.audio_data_root = audio_data_root
        self.device = device
        self.meta_path = config['meta_data_path']
        with open(self.meta_path) as f:
            self.meta = json.load(f)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):    
        meta_entry = self.meta[i]
        audio_path = meta_entry.get('path')
        caption_id = meta_entry.get('caption_id')
        audio_full_path = os.path.join(self.audio_data_root, audio_path)        
        example = {
            "text": meta_entry['caption'],
            "caption_id": caption_id,
            "audio_full_path": audio_full_path,
            "seconds_start": 0,
            "seconds_end": 2097152 / 44100,
        }
        return example
class CollateFunction:
    def __init__(self, condition_type):
        self.condition_type = condition_type
    def __call__(self, examples):
        caption_id = [example["caption_id"] for example in examples]
        prompt_texts = [example["text"] for example in examples]
        audio_full_path = [example["audio_full_path"] for example in examples]
        seconds_start = [example["seconds_start"] for example in examples]
        seconds_end = [example["seconds_end"] for example in examples]
        batch = {
            "audio_full_path": audio_full_path,
            "caption_id":caption_id,
            "prompt_texts": prompt_texts,
            "seconds_start": seconds_start,
            "seconds_end": seconds_end,
        }

        return batch
def load_audio_file(filename):
    try:
        # Use torchaudio to load the file regardless of format
        audio, in_sr = torchaudio.load(filename)
        
        # Resample if necessary
        if in_sr != 44100:
            resample_tf = T.Resample(in_sr, 44100)
            audio = resample_tf(audio)
        
        return audio

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except RuntimeError as e:
        print(f"Error: Unable to process file {filename}. {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while loading {filename}: {e}")
        return None
    
model = MusicGen.get_pretrained('facebook/musicgen-stereo-melody-large')
model.set_generation_params(duration=2097152/44100)  # generate 8 seconds.
generator = torch.Generator().manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
output_dir = config["output_dir"]
os.makedirs(output_dir, exist_ok=True)
dataset = AudioInversionDataset(
    config,
    audio_data_root=config["audio_data_dir"],
    device="cuda",
    )
# val_size = config["validation_num"]
# train_size = len(dataset) - val_size  # Remaining 20% for validation

# # Ensure consistent splitting
# _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
# DataLoader
val_collate_fn = CollateFunction(condition_type=[])
val_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=val_collate_fn,
    num_workers=1,
    pin_memory=True
)
melody_score = []
for i, batch in enumerate(tqdm(val_dataloader)):
    caption_id = batch["caption_id"]
    audio_full_path = batch["audio_full_path"][0]
    prompt_texts = batch["prompt_texts"]
    description_path = os.path.join(output_dir, "description.txt")
    with open(description_path, 'a') as file:
        file.write(f'{prompt_texts}\n')
     ## music continuation generation
    waveform, sr = torchaudio.load(audio_full_path)
    # trim the reference audio to the desired length
    waveform = waveform[..., :int(24 * sr)]  

    output = model.generate_continuation(
        prompt=waveform, 
        prompt_sample_rate=sr,
        descriptions=prompt_texts, 
    )

    result = output.detach().cpu().squeeze(0)
    
    file_path = os.path.join(output_dir, f"{caption_id[0]}.wav")
    
    torchaudio.save(uri=file_path, src=result, sample_rate=32000)
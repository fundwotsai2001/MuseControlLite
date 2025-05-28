import os
from config_stable_audio_inference import get_config
from tqdm import tqdm
config = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from utils.melody_extract import compute_melody
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torchaudio import transforms as T
from utils.stable_audio_dataset_utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
import json
import torch.nn.functional as F
from SDD_melody_acc import extract_melody_one_hot
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
        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )
        self.config = config
        self.audio_data_root = audio_data_root
        self.device = device
        self.meta_path = config['meta_data_path']
        with open(self.meta_path) as f:
            self.meta = json.load(f)
        self.meta_key = list(self.meta.keys())
        self.filtered_meta_key = []
        self.invalid_entries = []  # Store problematic keys for logging
        for key in self.meta_key:
            meta_entry = self.meta[key]
            audio_path = meta_entry.get('path')
            dynamics_path = meta_entry.get('dynamics_path')
            melody_path = meta_entry.get('melody_path')
            rhythm_path = meta_entry.get('rhythm_path')

            # Check if all required paths are valid
            if all([audio_path, dynamics_path, melody_path, rhythm_path]):
                # Check if the files exist
                if (os.path.exists(os.path.join(self.audio_data_root, audio_path)) and
                    os.path.exists(dynamics_path) and
                    os.path.exists(melody_path) and
                    os.path.exists(rhythm_path)):
                    self.filtered_meta_key.append(key)
                else:
                    self.invalid_entries.append(key)
            else:
                self.invalid_entries.append(key)
        # print("len(self.filtered_meta_key)", len(self.filtered_meta_key))
        # print("len(self.meta_key)", len(self.meta_key))
        # Print problematic entries
        if self.invalid_entries:
            print("The following entries are invalid and have been excluded:")
            for key in self.invalid_entries:
                print(f" - {key}: {self.meta[key]}")

    def __len__(self):
        return len(self.filtered_meta_key)

    def __getitem__(self, i):
        # Extract metadata
        meta_entry = self.meta[self.filtered_meta_key[i]]
        audio_path = meta_entry.get('path')
        audio_full_path = os.path.join(self.audio_data_root, audio_path)

        # Create example dictionary
        example = {
            "audio_full_path": audio_full_path,
            "text": meta_entry['rephrased_caption'],
        }
        return example
    
class CollateFunction:
    def __init__(self, condition_type, mode="train"):
        self.condition_type = condition_type
        self.mode = mode  # "train" or "val"
    def __call__(self, examples):
        prompt_texts = [example["text"] for example in examples]
        audio_full_path = [example["audio_full_path"] for example in examples]
           
        batch = {
            "audio_full_path": audio_full_path,
            "prompt_texts": prompt_texts,
        }

        return batch
model = MusicGen.get_pretrained('facebook/musicgen-stereo-melody-large')
model.set_generation_params(duration=30)  # generate 8 seconds.
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
val_collate_fn = CollateFunction(condition_type=[], mode="val")
val_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=val_collate_fn,
    num_workers=config["dataloader_num_workers"],
    pin_memory=True
)
melody_score = []
for i, batch in enumerate(tqdm(val_dataloader)):
    audio_full_path = batch["audio_full_path"]
    prompt_texts = batch["prompt_texts"]
    description_path = os.path.join(output_dir, "description.txt")
    with open(description_path, 'a') as file:
        file.write(f'{prompt_texts}\n')
    melody, sr = torchaudio.load(audio_full_path[0])
    # generates using the melody from the given audio and the provided descriptions.
    wav = model.generate_with_chroma(prompt_texts, melody[None].expand(1, -1, -1), sr)
    
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    output_path = os.path.join(output_dir, f'{i}')
    audio_write(output_path, wav[0].cpu(), model.sample_rate, strategy="loudness")
    gen_melody = extract_melody_one_hot(output_path+".wav")
    print("gen_melody", gen_melody.shape)
    melody_condition = extract_melody_one_hot(audio_full_path[0])
    print("melody_condition", melody_condition.shape)
    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # Plot the first chromagram
    im1 = axes[0].imshow(gen_melody, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
    axes[0].set_title('Chroma 1')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Chroma Features')
    fig.colorbar(im1, ax=axes[0], orientation='vertical')
    # Plot the second chromagram
    im2 = axes[1].imshow(melody_condition, aspect='auto', origin='lower', interpolation='nearest', cmap='plasma')
    axes[1].set_title('Chroma 2')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Chroma Features')
    fig.colorbar(im2, ax=axes[1], orientation='vertical')
    min_len = min(gen_melody.shape[1], melody_condition.shape[1])
    print("min_len", min_len)
    matches = ((gen_melody[:, :min_len] == melody_condition[:, :min_len]) & (gen_melody[:, :min_len] == 1)).sum()
    accuracy = matches / min_len
    print("accuracy", accuracy)
    melody_score.append(accuracy)
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"compare_melody_{i}.png"))
    plt.close()
print("melody_score", np.mean(melody_score))
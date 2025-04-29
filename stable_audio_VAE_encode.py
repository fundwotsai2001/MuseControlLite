import torch
import soundfile as sf
from diffusers import StableAudioPipeline
import torchaudio
from torchaudio import transforms as T
import os
from utils.stable_audio_dataset_utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
from torch.utils.data import Dataset, random_split, DataLoader
from MuseControlLite_inference_on_the_fly_all import load_audio_file
import json
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
class AudioInversionDataset(Dataset):
    def __init__(
        self,
        meta_path,
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
        self.audio_data_root = audio_data_root
        self.device = device
        self.meta_path = meta_path
        with open(self.meta_path) as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        # Extract metadata
        meta_entry = self.meta[i]
        audio_path = meta_entry.get('path')
        # Load audio
        audio_full_path = os.path.join(self.audio_data_root, audio_path)
        audio = load_audio_file(audio_full_path)

        # Apply augmentations and encoding
        if self.augs is not None:
            audio = self.augs(audio)
        audio = audio.clamp(-1, 1)
        if self.encoding is not None:
            audio = self.encoding(audio)

        # Create example dictionary
        example = {
            # "caption_id": caption_id,
            "audio_full_path": audio_path,
            "audio": audio,
            "text": meta_entry['Qwen_caption'],
            "seconds_start": 0,
            "seconds_end": 2097152 / 44100
        }
        return example
    
class CollateFunction:
    def __init__(self, condition_type, mode="train"):
        self.condition_type = condition_type
        self.mode = mode  # "train" or "val"
    def __call__(self, examples):
        audio = [example["audio"][:, :2097152] for example in examples]
        audio_full_path = [example["audio_full_path"] for example in examples]
        prompt_texts = [example["text"] for example in examples]
        seconds_start = [example["seconds_start"] for example in examples]
        seconds_end = [example["seconds_end"] for example in examples]
        audio = torch.stack(audio).to(memory_format=torch.contiguous_format).float()
        batch = {
            "audio_full_path": audio_full_path,
            "audio": audio,
            "prompt_texts": prompt_texts,
            "seconds_start": seconds_start,
            "seconds_end": seconds_end,
        }

        return batch
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable-audio VAE encode")
    parser.add_argument("--audio_folder", type=str, default="../mtg_full_47s", help="The audio folder path")
    parser.add_argument("--meta_path", type=str, default="./Qwen_caption.json", help="A list with dictionaries save in a json file")
    parser.add_argument("--latent_dir", type=str, default="./Jamendo_audio_47s_latent", help="The path to save the encoded latent")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    args = parser.parse_args()  # Parse the arguments
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    vae = pipe.vae.to("cuda")
    pipe = pipe.to("cuda")
    vae.eval()
    dataset = AudioInversionDataset(
            meta_path=args.meta_path,
            audio_data_root=args.audio_folder,
            device="cuda",
            )
    latent_dir = args.latent_dir
    collate_fn = CollateFunction(condition_type=None, mode="val")
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    # set the seed for generator
    generator = torch.Generator("cuda").manual_seed(0)
    os.makedirs(latent_dir, exist_ok=True)
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        with torch.no_grad():
            latents = vae.encode(batch['audio'].to(dtype=torch.float16).to("cuda")).latent_dist.sample()
        for i in range(latents.shape[0]):
            latent_path = os.path.join(latent_dir, batch['audio_full_path'][i].replace('.mp3', '.pth'))
            os.makedirs(os.path.dirname(latent_path), exist_ok=True)
            print("latent_path", latent_path)
            torch.save(latents[i], latent_path)
            

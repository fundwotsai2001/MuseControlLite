import torch
import soundfile as sf
from pipeline.stable_audio_multi_cfg_pipe import StableAudioPipeline
from diffusers.loaders import AttnProcsLayers
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
    StableAudioAttnProcessor2_0_rotary_double,
)
import torch.nn as nn
import torch.nn.functional as F
from stable_audio_train_new_melody import melody_extractor, CollateFunction, AudioInversionDataset
from safetensors.torch import load_file  # Import safetensors
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from config_stable_audio_inference import get_config
import argparse
from utils.dynamics_control_for_stable_audio import compute_dynamics
from sklearn.metrics import f1_score
import json
from utils.compute_everything_new import compute_melody
from stable_audio_dataset_utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
from madmom.features.downbeats import DBNDownBeatTrackingProcessor,RNNDownBeatProcessor
import random
from torchaudio import transforms as T
from tqdm import tqdm
import torchaudio
from SDD_melody_acc import extract_melody_one_hot

def load_audio_file(filename, target_sr=44100, target_samples=2097152):
    try:
        audio, in_sr = torchaudio.load(filename)    
        # Resample if necessary
        if in_sr != target_sr:
            resampler = T.Resample(in_sr, target_sr)
            audio = resampler(audio)
            
        augs = torch.nn.Sequential(
            PhaseFlipper(),
        )
        audio = augs(audio)
        audio = audio.clamp(-1, 1)
        encoding = torch.nn.Sequential(
            Stereo(),
        )
        audio = audio.mean(dim=0, keepdim=True)
        # audio = audio[:1,:]
        audio = encoding(audio)
        # audio.shape is [channels, samples]
        num_samples = audio.shape[-1]

        if num_samples < target_samples:
            # Pad if it's too short
            pad_amount = target_samples - num_samples
            # Zero-pad at the end (or randomly if you prefer)
            audio = F.pad(audio, (0, pad_amount)) 
            print(f"pad {pad_amount}")
        else:
            # max_start = num_samples - target_samples
            # start_idx = random.randint(0, max_start)
            # print(f"Sampling from index {start_idx} to {start_idx + target_samples}")
            # audio = audio[:, start_idx:start_idx + target_samples]
            audio = audio[:, :2097152]
        # Optional clamp
        
        return audio
    except RuntimeError:
        print(f"Failed to decode audio file: {filename}")
        return None
def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
    generator = torch.Generator().manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    condition_extractors = {}
    score_dynamics = []
    score_rhythm = []
    score_melody = []
    id = [49,49,49,322,322,322,610,610,785,785,933,933,1069,1069,57,703]
    if config["apadapter"]:
        melody_conditoner = melody_extractor().cuda().float()
        condition_extractors["melody"] = melody_conditoner
        melody_conditoner.eval()
        for conditioner in condition_extractors.values():
            conditioner.requires_grad_(True)
    for conditioner_type, ckpt_path in config["extractor_ckpt"].items():
        if "bin" in ckpt_path:
            state_dict = torch.load(ckpt_path)
        elif "safetensors" in ckpt_path:
            state_dict = load_file(ckpt_path, device="cpu")
        condition_extractors[conditioner_type].load_state_dict(state_dict)
        print(f"load checkpoint from {config['extractor_ckpt']} successfully !")
    output_dir = config["output_dir"] + f"_{config['guidance_scale_text']}_{config['guidance_scale_con']}_{config['guidance_scale_audio']}_{'_'.join(config['condition_type'])}"
    os.makedirs(output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if config["weight_dtype"] == "fp16":
        weight_dtype = torch.float16
    elif config["weight_dtype"] == "bp16":
        weight_dtype = torch.bfloat16
    print("weight_dtype", weight_dtype)
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
    pipe.scheduler.config.sigma_max = config["sigma_max"]
    pipe.scheduler.config.sigma_min = config["sigma_min"]
    transformer = pipe.transformer
    attn_procs = {}
    processor_classes = {
        "rotary": StableAudioAttnProcessor2_0_rotary,
        "rotary_double": StableAudioAttnProcessor2_0_rotary_double,
    }
    # Get the processor classes based on the type
    attn_processor = processor_classes.get(config["attn_processor_type"], None)
    for name in transformer.attn_processors.keys():
        if name.endswith("attn1.processor"):
            attn_procs[name] = StableAudioAttnProcessor2_0()
        else:
            attn_procs[name] = attn_processor(
                layer_id = name.split(".")[1],
                hidden_size=768,
                name=name,
                cross_attention_dim=768,
                scale=1.0,
            ).to("cuda", dtype=torch.float)
    if config["transformer_ckpt"] is not None:
        if "bin" in config["transformer_ckpt"]:
            state_dict = torch.load(config["transformer_ckpt"])
        elif "safetensors" in config["transformer_ckpt"]:
            state_dict = load_file(config["transformer_ckpt"], device="cuda")
        for name, processor in attn_procs.items():
            if isinstance(processor, attn_processor):
                weight_name_v = name + ".to_v_ip.weight"
                weight_name_k = name + ".to_k_ip.weight"
                conv_out_weight = name + ".conv_out.weight"
                processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].to(weight_dtype))
                processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].to(weight_dtype))
                processor.conv_out.weight = torch.nn.Parameter(state_dict[conv_out_weight].to(weight_dtype))
                print(f"load {name}")
    transformer.set_attn_processor(attn_procs)
    class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return pipe.transformer(*args, **kwargs)

    transformer = _Wrapper(pipe.transformer.attn_processors)
    transformer.eval()
    pipe = pipe.to("cuda")
    dataset = config["audio_files"]
    negative_text_prompt = config["negative_text_prompt"]
    mask_start = int(config["mask_start_seconds"] / 30 * 1024)
    mask_end = int(config["mask_end_seconds"] / 30 * 1024)
    print("mask_start:", mask_start)
    print("mask_end:", mask_end)
    dynamic_roll = []

    x_torch = torch.linspace(0, 2 * math.pi, steps=8280)
    roll_torch = 10 * torch.sin(x_torch + math.pi / 2) - 30
    dynamic_roll.append(roll_torch)

    x_torch = torch.linspace(-math.pi, math.pi, steps=8280)
    roll_torch = 10 * torch.sin(x_torch + math.pi / 2) - 30
    dynamic_roll.append(roll_torch)
    
    x_torch = torch.linspace(-20, -60, steps=8280)
    dynamic_roll.append(x_torch)

    x_torch = torch.linspace(-60, -20, steps=8280)
    dynamic_roll.append(x_torch)
    with torch.no_grad():
        for i, audio_file in enumerate(dataset):
            print(audio_file)
            prompt_texts = config["text"][i]
            if config["no_text"] is True:
                prompt_texts = negative_text_prompt
            description_path = os.path.join(output_dir, "description.txt")
            with open(description_path, 'a') as file:
                file.write(f'{prompt_texts}\n')
            if "melody" in config["condition_type"]:
                melody_condition = compute_melody(audio_file)
                melody_condition = torch.from_numpy(melody_condition).cuda().unsqueeze(0)
                # melody_condition = F.interpolate(melody_condition, size=1296, mode='linear', align_corners=False)
                extracted_melody_condition = condition_extractors["melody"](melody_condition.float()) 
                masked_extracted_melody_condition = torch.full_like(extracted_melody_condition.to(weight_dtype), fill_value=0)
                # extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                # masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                # extracted_melody_condition[: ,:, : mask_start] = 0
                # extracted_melody_condition[: ,:, mask_end:] = 0
                
            else: 
                extracted_melody_condition = torch.full((1, 768, 1024), 0, device='cuda')
                masked_extracted_melody_condition = extracted_melody_condition
                ### concat conditions
            # extracted_condition = torch.concat((extracted_rhythm_condition, extracted_dynamics_condition, extracted_melody_condition), dim=1)
            # masked_extracted_condition = torch.concat((masked_extracted_rhythm_condition, masked_extracted_dynamics_condition, masked_extracted_melody_condition), dim=1).float()
            extracted_condition = torch.concat((masked_extracted_melody_condition, masked_extracted_melody_condition, extracted_melody_condition), dim=0).float()
            extracted_condition = extracted_condition.transpose(1, 2)
            print(prompt_texts[0])
            waveform = pipe(
                # extracted_condition_audio = extracted_condition_audio,
                extracted_condition = extracted_condition, 
                prompt=prompt_texts[0],
                negative_prompt=negative_text_prompt,
                num_inference_steps=config["denoise_step"],
                guidance_scale_text=config["guidance_scale_text"],
                guidance_scale_con=config["guidance_scale_con"],
                # guidance_scale_audio = config["guidance_scale_audio"],
                num_waveforms_per_prompt=1,
                audio_end_in_s= 2097152 / 44100,
                generator=generator,
            ).audios 
            print(f"{i}")       
            gen_file_path = os.path.join(output_dir, f"{id[i]}_{i}.wav")
            output = waveform[0].T.float().cpu().numpy()
            sf.write(gen_file_path, output, pipe.vae.sampling_rate)
            original_path = os.path.join(output_dir, f"original_{i}.wav")
            audio = load_audio_file(audio_file)
            original_audio = audio.T.float().cpu().numpy()
            sf.write(original_path, original_audio, pipe.vae.sampling_rate)

            melody_condition = extract_melody_one_hot(audio_file)      
            gen_melody = extract_melody_one_hot(gen_file_path)
            min_len_melody = min(gen_melody.shape[1], melody_condition.shape[1])
            matches = ((gen_melody[:, :min_len_melody] == melody_condition[:, :min_len_melody]) & (gen_melody[:, :min_len_melody] == 1)).sum()
            accuracy = matches / min_len_melody
            score_melody.append(accuracy)
            print("melody accuracy", accuracy)
            # Adjust layout to avoid overlap
            

            fig, axes = plt.subplots(2, 1, figsize=(18, 10))
            # For the first subplot:
            ax = axes[0]
            im2 = ax.imshow(melody_condition[:, :min_len_melody], aspect='auto', origin='lower',
                            interpolation='nearest', cmap='plasma')
            ax.set_title('Melody Condition')
            ax.set_xlabel('Time')
            ax.set_ylabel('Chroma Features')

            # For the second subplot:
            ax = axes[1]
            im1 = ax.imshow(gen_melody[:, :min_len_melody], aspect='auto', origin='lower',
                            interpolation='nearest', cmap='viridis')
            ax.set_title('Generated Melody')
            ax.set_xlabel('Time')
            ax.set_ylabel('Chroma Features')


        data_to_save = {"config": config}

        # if "melody" in config["condition_type"]:
        data_to_save["score_melody"] = np.mean(score_melody)
        print(data_to_save)
        file_path = os.path.join(output_dir, "result.txt")
        with open(file_path, "w") as file:
            json.dump(data_to_save, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AP-adapter Inference Script")
    config = get_config()  # Pass the parsed arguments to get_config
    main(config)

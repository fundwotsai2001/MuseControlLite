import torch
import soundfile as sf
from diffusers.loaders import AttnProcsLayers
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
    StableAudioAttnProcessor2_0_rotary_double,
)
import torch.nn as nn
import torch.nn.functional as F
from MuseControlLite_train_melody_only import melody_extractor_full
from safetensors.torch import load_file  # Import safetensors
import os
import numpy as np
import matplotlib.pyplot as plt
from config_inference import get_config
import argparse
import json
from utils.extract_conditions import compute_melody, compute_dynamics, extract_melody_one_hot, evaluate_f1_rhythm
from utils.stable_audio_dataset_utils import Stereo, PhaseFlipper
from madmom.features.downbeats import DBNDownBeatTrackingProcessor,RNNDownBeatProcessor
import random
from torchaudio import transforms as T
from tqdm import tqdm
import torchaudio
from madmom.features.downbeats import DBNDownBeatTrackingProcessor,RNNDownBeatProcessor
from torch.utils.data import Dataset, random_split, DataLoader
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
        # Build file paths
        def build_path(root, path, ext_in='.mp3', ext_out='.npy'):
            file_name = path.replace("/", "_").replace(ext_in, ext_out)
            return os.path.join(root, file_name)
        # Load numpy arrays concurrently
        def load_npy(path):
            return np.load(path) 
        melody_path = build_path("./SDD_nosinging_audio_conditions/SDD_melody_condition_dir", audio_path)
        melody_curve = load_npy(melody_path)
        
        # Load audio tokens, they are encoded with the Stable-audio VAE and saved, skipping the the VAE encoding process saves memory when training MuseControlLite
        audio_full_path = os.path.join(self.audio_data_root, audio_path)        
        example = {
            "text": meta_entry['caption'],
            "caption_id": caption_id,
            "audio_full_path": audio_full_path,
            "melody_curve": melody_curve,
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
        if len(self.condition_type) != 0:
            melody_condition = [example["melody_curve"] for example in examples]
            melody_condition = [torch.tensor(cond) for cond in melody_condition]
            melody_condition = torch.stack(melody_condition)
            batch = {
                "caption_id":caption_id,
                "audio_full_path": audio_full_path,
                "melody_condition": melody_condition,
                "prompt_texts": prompt_texts,
                "seconds_start": seconds_start,
                "seconds_end": seconds_end,
            }
        else:
            audio = torch.stack(audio).to(memory_format=torch.contiguous_format).float()
            batch = {
                # "audio_full_path": audio_full_path,
                "caption_id":caption_id,
                "audio": audio,
                "prompt_texts": prompt_texts,
                "seconds_start": seconds_start,
                "seconds_end": seconds_end,
            }

        return batch
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
    if config["apadapter"]:
        melody_conditoner = melody_extractor_full().cuda().float()
        condition_extractors["melody"] = melody_conditoner
        melody_conditoner.eval()
        for conditioner_type, ckpt_path in config["extractor_ckpt_melody"].items():
            if "bin" in ckpt_path:
                state_dict = torch.load(ckpt_path)
            elif "safetensors" in ckpt_path:
                state_dict = load_file(ckpt_path, device="cpu")
            condition_extractors[conditioner_type].load_state_dict(state_dict)
            print(f"load checkpoint from {config['extractor_ckpt_melody']} successfully !")
    output_dir = config["output_dir"] + f"text_{config['guidance_scale_text']}_con_{config['guidance_scale_con']}_{'_'.join(config['condition_type'])}_{config['sigma_min']}_{config['sigma_max']}_step{config['denoise_step']}"
    os.makedirs(output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if config["weight_dtype"] == "fp16":
        weight_dtype = torch.float16
    elif config["weight_dtype"] == "bp16":
        weight_dtype = torch.bfloat16
    if config["apadapter"]:
        from pipeline.stable_audio_multi_cfg_pipe import StableAudioPipeline
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
                    scale=config['ap_scale'],
                ).to("cuda", dtype=torch.float)
        if config["transformer_ckpt_melody"] is not None:
            if "bin" in config["transformer_ckpt"]:
                state_dict = torch.load(config["transformer_ckpt_melody"])
            elif "safetensors" in config["transformer_ckpt_melody"]:
                state_dict = load_file(config["transformer_ckpt_melody"], device="cuda")
            for name, processor in attn_procs.items():
                if isinstance(processor, attn_processor):
                    weight_name_v = name + ".to_v_ip.weight"
                    weight_name_k = name + ".to_k_ip.weight"
                    conv_out_weight = name + ".conv_out.weight"
                    processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].to(torch.float32))
                    processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].to(torch.float32))
                    processor.conv_out.weight = torch.nn.Parameter(state_dict[conv_out_weight].to(torch.float32))
                    print(f"load {name}")
        transformer.set_attn_processor(attn_procs)
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipe.transformer(*args, **kwargs)
        transformer = _Wrapper(pipe.transformer.attn_processors)
    else:
        from diffusers import StableAudioPipeline
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
        pipe.scheduler.config.sigma_max = config["sigma_max"]
        pipe.scheduler.config.sigma_min = config["sigma_min"]
    dataset = AudioInversionDataset(
        config,
        audio_data_root=config["audio_data_dir"],
        device="cuda",
        )
    val_collate_fn = CollateFunction(condition_type=config['condition_type'])
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=val_collate_fn,
        num_workers=1,
        pin_memory=True
    )
    pipe = pipe.to("cuda")
    negative_text_prompt = config["negative_text_prompt"]

    # Apply masks for audio condition and musical attribute condition, the masked parts will be assign to zero, sames are the drop condition in cfg.
    total_seconds = 2097152/44100
    if config['use_audio_mask']:
        audio_mask_start = int(config["audio_mask_start_seconds"] / total_seconds * 1024) # 1024 is the latent length for 2097152/44100 seconds
        audio_mask_end = int(config["audio_mask_end_seconds"] / total_seconds * 1024)
    elif config['use_musical_attribute_mask']:
        musical_attribute_mask_start = int(config["musical_attribute_mask_start_seconds"] / total_seconds * 1024)
        musical_attribute_mask_end = int(config["musical_attribute_mask_end_seconds"] / total_seconds * 1024)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            if config["apadapter"]:
                prompt_texts = batch["prompt_texts"]
                caption_id = batch["caption_id"]
                audio_full_path = batch['audio_full_path']
                description_path = os.path.join(output_dir, "description.txt")
                with open(description_path, 'a') as file:
                    file.write(f'{prompt_texts}\n')
                # For single conition, we can utilize the full cross-attention dimension 768, instead of 768/4 in MuseControlLite_inference_on_the_fly_all.py
                if "melody" in config["condition_type"]:
                    melody_condition = batch['melody_condition'].cuda()
                    # melody_condition = torch.from_numpy(melody_condition).cuda()
                    print("melody_condition", melody_condition.shape)
                    extracted_melody_condition = condition_extractors["melody"](melody_condition.to(torch.float32))
                    masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
                    extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                    masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                else: 
                    extracted_melody_condition = torch.zeros((1, 768, 1024), device="cuda")
                    masked_extracted_melody_condition = extracted_melody_condition
                if config['use_musical_attribute_mask']:
                    extracted_melody_condition[:,:,musical_attribute_mask_start:musical_attribute_mask_end] = 0
                # Use multiple cfg
                extracted_condition = torch.concat((masked_extracted_melody_condition, masked_extracted_melody_condition, extracted_melody_condition), dim=0)
                extracted_condition = extracted_condition.transpose(1, 2)
                waveform = pipe(
                    extracted_condition = extracted_condition, 
                    prompt=prompt_texts,
                    negative_prompt=negative_text_prompt,
                    num_inference_steps=config["denoise_step"],
                    guidance_scale_text=config["guidance_scale_text"],
                    guidance_scale_con=config["guidance_scale_con"],
                    num_waveforms_per_prompt=1,
                    audio_end_in_s=2097152 / 44100,
                    generator=generator,
                ).audios 
                print(f"{i}")       
                gen_file_path = os.path.join(output_dir, f"{caption_id[0]}.wav")
                output = waveform[0].T.float().cpu().numpy()
                sf.write(gen_file_path, output, pipe.vae.sampling_rate)
                # original_path = os.path.join(output_dir, f"original_{i}.wav")
                # original_audio = audio.T.float().cpu().numpy()
                # sf.write(original_path, original_audio, pipe.vae.sampling_rate)
                audio_full_path = audio_full_path[0]

                # Dynamics correlation evaluation
                # dynamics_condition = compute_dynamics(audio_full_path)
                # gen_dynamics = compute_dynamics(gen_file_path)
                # min_len_dynamics = min(gen_dynamics.shape[0], dynamics_condition.shape[0])
                # pearson_corr = np.corrcoef(gen_dynamics[:min_len_dynamics], dynamics_condition[:min_len_dynamics])[0, 1]
                # print("pearson_corr", pearson_corr)
                # score_dynamics.append(pearson_corr)

                # Melody accuracy evaluation
                melody_condition = extract_melody_one_hot(audio_full_path)      
                gen_melody = extract_melody_one_hot(gen_file_path)
                min_len_melody = min(gen_melody.shape[1], melody_condition.shape[1])
                matches = ((gen_melody[:, :min_len_melody] == melody_condition[:, :min_len_melody]) & (gen_melody[:, :min_len_melody] == 1)).sum()
                accuracy = matches / min_len_melody
                score_melody.append(accuracy)
                print("melody accuracy", accuracy)

                # # Beat detection f1 
                # # Adjust layout to avoid overlap
                # processor = RNNDownBeatProcessor()
                # input_probabilities = processor(audio_full_path)
                # generated_probabilities = processor(gen_file_path)
                # hmm_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)
                # input_timestamps = hmm_processor(input_probabilities)
                # generated_timestamps = hmm_processor(generated_probabilities)
                # precision, recall, f1 = evaluate_f1_rhythm(input_timestamps, generated_timestamps)
                # # Output results
                # print(f"F1 Score: {f1:.2f}")
                # score_rhythm.append(f1)

                # # Plotting
                # frame_rate = 100  # Frames per second
                # input_time_axis = np.linspace(0, len(input_probabilities) / frame_rate, len(input_probabilities))
                # generate_time_axis = np.linspace(0, len(generated_probabilities) / frame_rate, len(generated_probabilities))
                # fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust figsize as needed

                # # ----------------------------
                # # Subplot (0,0): Dynamics Plot
                # ax = axes[0, 0]
                # ax.plot(dynamics_condition[:min_len_dynamics].squeeze(), linewidth=1, label='Dynamics condition')
                # ax.set_title('Dynamics')
                # ax.set_xlabel('Time Frame')
                # ax.set_ylabel('Dynamics (dB)')
                # ax.legend(fontsize=8)
                # ax.grid(True)
                # # ----------------------------
                # # Subplot (0,0): Dynamics Plot
                # ax = axes[1, 0]
                # ax.plot(gen_dynamics[:min_len_dynamics].squeeze(), linewidth=1, label='Generated Dynamics')
                # ax.set_title('Dynamics')
                # ax.set_xlabel('Time Frame')
                # ax.set_ylabel('Dynamics (dB)')
                # ax.legend(fontsize=8)
                # ax.grid(True)

                # # ----------------------------
                # # Subplot (0,2): Melody Condition (Chromagram)
                # ax = axes[0, 1]
                # im2 = ax.imshow(melody_condition[:, :min_len_melody], aspect='auto', origin='lower',
                #                 interpolation='nearest', cmap='plasma')
                # ax.set_title('Melody Condition')
                # ax.set_xlabel('Time')
                # ax.set_ylabel('Chroma Features')

                # # ----------------------------
                # # Subplot (0,1): Generated Melody (Chromagram)
                # ax = axes[1, 1]
                # im1 = ax.imshow(gen_melody[:, :min_len_melody], aspect='auto', origin='lower',
                #                 interpolation='nearest', cmap='viridis')
                # ax.set_title('Generated Melody')
                # ax.set_xlabel('Time')
                # ax.set_ylabel('Chroma Features')

                # # ----------------------------
                # # Subplot (1,0): Rhythm Input Probabilities
                # ax = axes[0, 2]
                # ax.plot(input_time_axis, input_probabilities,
                #         label="Input Beat Probability")
                # ax.plot(input_time_axis, input_probabilities,
                #         label="Input Downbeat Probability", alpha=0.8)
                # ax.set_title('Rhythm: Input')
                # ax.set_xlabel('Time (s)')
                # ax.set_ylabel('Probability')
                # ax.legend()
                # ax.grid(True)

                # # ----------------------------
                # # Subplot (1,1): Rhythm Generated Probabilities
                # ax = axes[1, 2]
                # ax.plot(generate_time_axis, generated_probabilities,
                #         color='orange', label="Generated Beat Probability")
                # ax.plot(generate_time_axis, generated_probabilities,
                #         alpha=0.8, color='red', label="Generated Downbeat Probability")
                # ax.set_title('Rhythm: Generated')
                # ax.set_xlabel('Time (s)')
                # ax.set_ylabel('Probability')
                # ax.legend()
                # ax.grid(True)


                # # Adjust layout and save the combined image
                # plt.tight_layout()
                # combined_path = os.path.join(output_dir, f"combined_{i}.png")
                # plt.savefig(combined_path)
                # plt.close()

                # print(f"Combined plot saved to {combined_path}")   
            else:
                audio = pipe(
                    prompt=prompt_texts,
                    negative_prompt=negative_text_prompt,
                    num_inference_steps=config["denoise_step"],
                    guidance_scale=config["guidance_scale_text"],
                    num_waveforms_per_prompt=1,
                    audio_end_in_s=2097152/44100,
                    generator=generator,
                ).audios
                output = audio[0].T.float().cpu().numpy()
                file_path = os.path.join(output_dir, f"{prompt_texts}.wav")
                sf.write(file_path, output, pipe.vae.sampling_rate)         
        data_to_save = {"config": config}

        # if "dynamics" in config["condition_type"]:
        data_to_save["score_dynamics"] = np.mean(score_dynamics)

        # if "rhythm" in config["condition_type"]:
        data_to_save["score_rhythm"] = np.mean(score_rhythm)

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

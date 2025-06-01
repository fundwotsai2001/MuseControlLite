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
from MuseControlLite_train_all import dynamics_extractor, rhythm_extractor, melody_extractor
from safetensors.torch import load_file  # Import safetensors
import os
import numpy as np
import matplotlib.pyplot as plt
from config_inference_infilling import get_config
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
            audio = audio[:, :target_samples]
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
    if config["apadapter"]:
        dynamics_conditoner = dynamics_extractor().cuda().float()
        condition_extractors["dynamics"] = dynamics_conditoner
        dynamics_conditoner.eval()    
        rhythm_conditoner = rhythm_extractor().cuda().float()
        condition_extractors["rhythm"] = rhythm_conditoner
        rhythm_conditoner.eval()
        melody_conditoner = melody_extractor().cuda().float()
        condition_extractors["melody"] = melody_conditoner
        melody_conditoner.eval()
        for conditioner_type, ckpt_path in config["extractor_ckpt"].items():
            if "bin" in ckpt_path:
                state_dict = torch.load(ckpt_path)
            elif "safetensors" in ckpt_path:
                state_dict = load_file(ckpt_path, device="cpu")
            condition_extractors[conditioner_type].load_state_dict(state_dict)
            print(f"load checkpoint from {config['extractor_ckpt']} successfully !")
    output_dir = config["output_dir"] + f"text_{config['guidance_scale_text']}_con_{config['guidance_scale_con']}_{'_'.join(config['condition_type'])}_{config['sigma_min']}_{config['sigma_max']}"
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
        for i, prompt_texts in enumerate(config['text']):
            if config["apadapter"]:
                audio_file = config["audio_files"][i]
                if config["no_text"] is True:
                    prompt_texts = ""
                description_path = os.path.join(output_dir, "description.txt")
                with open(description_path, 'a') as file:
                    file.write(f'{prompt_texts}\n')
                if "dynamics" in config["condition_type"]:
                    dynamics_condition = compute_dynamics(audio_file)
                    dynamics_condition = torch.from_numpy(dynamics_condition).cuda()
                    dynamics_condition = dynamics_condition.unsqueeze(0).unsqueeze(0)
                    print("dynamics_condition", dynamics_condition.shape)
                    extracted_dynamics_condition = condition_extractors["dynamics"](dynamics_condition.to(torch.float32))
                    masked_extracted_dynamics_condition =  torch.zeros_like(extracted_dynamics_condition)
                    extracted_dynamics_condition = F.interpolate(extracted_dynamics_condition, size=1024, mode='linear', align_corners=False) 
                    masked_extracted_dynamics_condition = F.interpolate(masked_extracted_dynamics_condition, size=1024, mode='linear', align_corners=False)
                else: 
                    extracted_dynamics_condition = torch.zeros((1, 192, 1024), device="cuda")
                    masked_extracted_dynamics_condition = extracted_dynamics_condition
                if "rhythm" in config["condition_type"]:
                    rnn_processor = RNNDownBeatProcessor()
                    wave = load_audio_file(audio_file)
                    original_path = os.path.join(output_dir, f"original_{i}.wav")
                    sf.write(original_path, wave.T.float().cpu().numpy(), 44100)
                    rhythm_curve = rnn_processor(original_path)
                    rhythm_condition = torch.from_numpy(rhythm_curve).cuda()
                    rhythm_condition = rhythm_condition.transpose(0,1).unsqueeze(0)
                    print("rhythm_condition", rhythm_condition.shape)
                    extracted_rhythm_condition = condition_extractors["rhythm"](rhythm_condition.to(torch.float32))
                    masked_extracted_rhythm_condition = torch.zeros_like(extracted_rhythm_condition)
                    extracted_rhythm_condition = F.interpolate(extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)
                    masked_extracted_rhythm_condition = F.interpolate(masked_extracted_rhythm_condition, size=1024, mode='linear', align_corners=False)      
                else: 
                    extracted_rhythm_condition = torch.zeros((1, 192, 1024), device="cuda")
                    masked_extracted_rhythm_condition = extracted_rhythm_condition
                if "melody" in config["condition_type"]:
                    melody_condition = compute_melody(audio_file)
                    melody_condition = torch.from_numpy(melody_condition).cuda().unsqueeze(0)
                    print("melody_condition", melody_condition.shape)
                    extracted_melody_condition = condition_extractors["melody"](melody_condition.to(torch.float32))
                    masked_extracted_melody_condition = torch.zeros_like(extracted_melody_condition)
                    extracted_melody_condition = F.interpolate(extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                    masked_extracted_melody_condition = F.interpolate(masked_extracted_melody_condition, size=1024, mode='linear', align_corners=False)
                else: 
                    extracted_melody_condition = torch.zeros((1, 192, 1024), device="cuda")
                    masked_extracted_melody_condition = extracted_melody_condition
                if "audio" in config["condition_type"]:
                    desired_repeats = 192 // 64  # Number of repeats needed
                    audio = load_audio_file(audio_file)
                    audio_condition = pipe.vae.encode(audio.unsqueeze(0).to(weight_dtype).cuda()).latent_dist.sample()
                    extracted_audio_condition = audio_condition.repeat_interleave(desired_repeats, dim=1).float()
                    masked_extracted_audio_condition = torch.zeros_like(extracted_audio_condition)
                else: 
                    extracted_audio_condition = torch.zeros((1, 192, 1024), device="cuda")
                    masked_extracted_audio_condition = extracted_audio_condition
                if config['use_audio_mask']:
                    extracted_rhythm_condition[:,:,:audio_mask_start] = 0
                    extracted_rhythm_condition[:,:,audio_mask_end:] = 0
                    extracted_dynamics_condition[:,:,:audio_mask_start] = 0
                    extracted_dynamics_condition[:,:,audio_mask_end:] = 0
                    extracted_melody_condition[:,:,:audio_mask_start] = 0
                    extracted_melody_condition[:,:,audio_mask_end:] = 0
                    extracted_audio_condition[:,:,audio_mask_start:audio_mask_end] = 0
                elif config['use_musical_attribute_mask']:
                    extracted_rhythm_condition[:,:,musical_attribute_mask_start:musical_attribute_mask_end] = 0
                    extracted_dynamics_condition[:,:,musical_attribute_mask_start:musical_attribute_mask_end] = 0
                    extracted_melody_condition[:,:,musical_attribute_mask_start:musical_attribute_mask_end] = 0
                    extracted_audio_condition[:,:,:musical_attribute_mask_start] = 0
                    extracted_audio_condition[:,:,musical_attribute_mask_end:] = 0
                # print("extracted_rhythm_condition, extracted_dynamics_condition, extracted_melody_condition, extracted_audio_condition", extracted_rhythm_condition.dtype, extracted_dynamics_condition.dtype, extracted_melody_condition.dtype, extracted_audio_condition.dtype)
                # Use multiple cfg
                extracted_condition = torch.concat((extracted_rhythm_condition, extracted_dynamics_condition, extracted_melody_condition, extracted_audio_condition), dim=1)
                masked_extracted_condition = torch.concat((masked_extracted_rhythm_condition, masked_extracted_dynamics_condition, masked_extracted_melody_condition, masked_extracted_audio_condition), dim=1)
                extracted_condition = torch.concat((masked_extracted_condition, masked_extracted_condition, extracted_condition), dim=0)
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
                gen_file_path = os.path.join(output_dir, f"test_{i}.wav")
                output = waveform[0].T.float().cpu().numpy()
                sf.write(gen_file_path, output, pipe.vae.sampling_rate)
                original_path = os.path.join(output_dir, f"original_{i}.wav")
                audio = load_audio_file(audio_file)
                original_audio = audio.T.float().cpu().numpy()
                sf.write(original_path, original_audio, pipe.vae.sampling_rate)
                dynamics_condition = compute_dynamics(audio_file)
                gen_dynamics = compute_dynamics(gen_file_path)
                min_len_dynamics = min(gen_dynamics.shape[0], dynamics_condition.shape[0])
                pearson_corr = np.corrcoef(gen_dynamics[:min_len_dynamics], dynamics_condition[:min_len_dynamics])[0, 1]
                print("pearson_corr", pearson_corr)
                score_dynamics.append(pearson_corr)
                melody_condition = extract_melody_one_hot(audio_file)      
                gen_melody = extract_melody_one_hot(gen_file_path)
                min_len_melody = min(gen_melody.shape[1], melody_condition.shape[1])
                matches = ((gen_melody[:, :min_len_melody] == melody_condition[:, :min_len_melody]) & (gen_melody[:, :min_len_melody] == 1)).sum()
                accuracy = matches / min_len_melody
                score_melody.append(accuracy)
                print("melody accuracy", accuracy)
                # Adjust layout to avoid overlap
                processor = RNNDownBeatProcessor()
                input_probabilities = processor(audio_file)
                generated_probabilities = processor(gen_file_path)
                hmm_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)
                input_timestamps = hmm_processor(input_probabilities)
                generated_timestamps = hmm_processor(generated_probabilities)
                precision, recall, f1 = evaluate_f1_rhythm(input_timestamps, generated_timestamps)
                # Output results
                print(f"F1 Score: {f1:.2f}")
                score_rhythm.append(f1)
                # Plotting
                frame_rate = 100  # Frames per second
                input_time_axis = np.linspace(0, len(input_probabilities) / frame_rate, len(input_probabilities))
                generate_time_axis = np.linspace(0, len(generated_probabilities) / frame_rate, len(generated_probabilities))
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust figsize as needed

                # ----------------------------
                # Subplot (0,0): Dynamics Plot
                ax = axes[0, 0]
                ax.plot(dynamics_condition[:min_len_dynamics].squeeze(), linewidth=1, label='Dynamics condition')
                ax.set_title('Dynamics')
                ax.set_xlabel('Time Frame')
                ax.set_ylabel('Dynamics (dB)')
                ax.legend(fontsize=8)
                ax.grid(True)
                # ----------------------------
                # Subplot (0,0): Dynamics Plot
                ax = axes[1, 0]
                ax.plot(gen_dynamics[:min_len_dynamics].squeeze(), linewidth=1, label='Generated Dynamics')
                ax.set_title('Dynamics')
                ax.set_xlabel('Time Frame')
                ax.set_ylabel('Dynamics (dB)')
                ax.legend(fontsize=8)
                ax.grid(True)

                # ----------------------------
                # Subplot (0,2): Melody Condition (Chromagram)
                ax = axes[0, 1]
                im2 = ax.imshow(melody_condition[:, :min_len_melody], aspect='auto', origin='lower',
                                interpolation='nearest', cmap='plasma')
                ax.set_title('Melody Condition')
                ax.set_xlabel('Time')
                ax.set_ylabel('Chroma Features')

                # ----------------------------
                # Subplot (0,1): Generated Melody (Chromagram)
                ax = axes[1, 1]
                im1 = ax.imshow(gen_melody[:, :min_len_melody], aspect='auto', origin='lower',
                                interpolation='nearest', cmap='viridis')
                ax.set_title('Generated Melody')
                ax.set_xlabel('Time')
                ax.set_ylabel('Chroma Features')

                # ----------------------------
                # Subplot (1,0): Rhythm Input Probabilities
                ax = axes[0, 2]
                ax.plot(input_time_axis, input_probabilities,
                        label="Input Beat Probability")
                ax.plot(input_time_axis, input_probabilities,
                        label="Input Downbeat Probability", alpha=0.8)
                ax.set_title('Rhythm: Input')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Probability')
                ax.legend()
                ax.grid(True)

                # ----------------------------
                # Subplot (1,1): Rhythm Generated Probabilities
                ax = axes[1, 2]
                ax.plot(generate_time_axis, generated_probabilities,
                        color='orange', label="Generated Beat Probability")
                ax.plot(generate_time_axis, generated_probabilities,
                        alpha=0.8, color='red', label="Generated Downbeat Probability")
                ax.set_title('Rhythm: Generated')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Probability')
                ax.legend()
                ax.grid(True)


                # Adjust layout and save the combined image
                plt.tight_layout()
                combined_path = os.path.join(output_dir, f"combined_{i}.png")
                plt.savefig(combined_path)
                plt.close()

                print(f"Combined plot saved to {combined_path}")   
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

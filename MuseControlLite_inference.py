import torch
import soundfile as sf
from MuseControlLite_setup import (
    setup_MuseControlLite,
    initialize_condition_extractors,
    evaluate_and_plot_results,
    load_audio_file,
    process_musical_conditions
)
import os
import numpy as np
from config_inference import get_config
import argparse
import json

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if config["weight_dtype"] == "fp16":
        weight_dtype = torch.float16
    if config["apadapter"]:
        condition_extractors, transformer_ckpt = initialize_condition_extractors(config)
        MuseControlLite = setup_MuseControlLite(config, weight_dtype, transformer_ckpt)
        MuseControlLite = MuseControlLite.to("cuda")
    else:
        from diffusers import StableAudioPipeline
        stable_audio = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
        stable_audio = stable_audio.to("cuda")
    negative_text_prompt = config["negative_text_prompt"]
    # Apply masks for audio condition and musical attribute condition, the masked parts will be assign to zero, sames are the drop condition in cfg.
    score_dynamics = []
    score_rhythm = []
    score_melody = []
    with torch.no_grad():
        for i, prompt_texts in enumerate(config['text']):
            if config["apadapter"]:
                audio_file = config["audio_files"][i]
                description_path = os.path.join(output_dir, "description.txt")
                with open(description_path, 'a') as file:
                    file.write(f'{prompt_texts}\n')
                final_condition, final_condition_audio = process_musical_conditions(config, audio_file, condition_extractors, output_dir, i, weight_dtype, MuseControlLite)
                if config["no_text"] is True:
                    prompt_texts = ""
                print("prompt_texts", prompt_texts)
                waveform = MuseControlLite(
                    extracted_condition=final_condition, 
                    extracted_condition_audio=final_condition_audio,
                    prompt=prompt_texts,
                    negative_prompt=negative_text_prompt,
                    num_inference_steps=config["denoise_step"],
                    guidance_scale_text=config["guidance_scale_text"],
                    guidance_scale_con=config["guidance_scale_con"],
                    guidance_scale_audio=config["guidance_scale_audio"],
                    num_waveforms_per_prompt=1,
                    audio_end_in_s=2097152 / 44100,
                    generator = torch.Generator().manual_seed(42)
                ).audios 
                # save audio
                gen_file_path = os.path.join(output_dir, f"test_{i}.wav")
                output = waveform[0].T.float().cpu().numpy()
                sf.write(gen_file_path, output, MuseControlLite.vae.sampling_rate)
                original_path = os.path.join(output_dir, f"original_{i}.wav")
                audio = load_audio_file(audio_file)
                if audio is not None:
                    original_audio = audio.T.float().cpu().numpy()
                    sf.write(original_path, original_audio, MuseControlLite.vae.sampling_rate)
                if config['show_result_and_plt']:
                    dynamics_score, rhythm_score, melody_score = evaluate_and_plot_results(
                        audio_file, gen_file_path, output_dir, i
                    )
                    score_dynamics.append(dynamics_score)
                    score_rhythm.append(rhythm_score)
                    score_melody.append(melody_score)
            else:
                audio = stable_audio(
                    prompt=prompt_texts,
                    negative_prompt=negative_text_prompt,
                    num_inference_steps=config["denoise_step"],
                    guidance_scale=config["guidance_scale_text"],
                    num_waveforms_per_prompt=1,
                    audio_end_in_s=2097152/44100,
                    generator = torch.Generator().manual_seed(42)
                ).audios
                output = audio[0].T.float().cpu().numpy()
                file_path = os.path.join(output_dir, f"{prompt_texts}.wav")
                sf.write(file_path, output, stable_audio.vae.sampling_rate)         
        data_to_save = {"config": config}
        if config['show_result_and_plt']:
            data_to_save["score_dynamics"] = np.mean(score_dynamics)
            data_to_save["score_rhythm"] = np.mean(score_rhythm)
            data_to_save["score_melody"] = np.mean(score_melody)
        file_path = os.path.join(output_dir, "result.txt")
        with open(file_path, "w") as file:
            json.dump(data_to_save, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AP-adapter Inference Script")
    config = get_config()  # Pass the parsed arguments to get_config
    main(config)

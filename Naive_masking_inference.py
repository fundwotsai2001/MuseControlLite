## naive masking approach using Stable Audio

import argparse
import torch
import torchaudio
from audio_infilling_baseline.Naive_Masking.pipeline_audio_infilling import AudioInfillingPipeline
from MuseControlLite_inference_musical_ablation_SDD import CollateFunction, AudioInversionDataset
from config_inference_infilling import get_config
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
import soundfile as sf
import os
from utils.extract_conditions import compute_melody, compute_dynamics, extract_melody_one_hot, evaluate_f1_rhythm
from madmom.features.downbeats import DBNDownBeatTrackingProcessor,RNNDownBeatProcessor
import numpy as np
import json
def main():
    config = get_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = AudioInfillingPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    dataset = AudioInversionDataset(
        config,
        audio_data_root=config["audio_data_dir"],
        device="cuda",
        )
    val_collate_fn = CollateFunction(condition_type=config["condition_type"])
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=val_collate_fn,
        num_workers=1,
        pin_memory=True
    )
    generator = torch.Generator("cuda").manual_seed(0)
    negative_text_prompt = config["negative_text_prompt"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    score_dynamics = []
    score_rhythm = []
    score_melody = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            # if i >= config["test_num"]:
            #     break
            caption_id = batch["caption_id"]
            prompt_texts = batch["prompt_texts"]
            audio_full_path = batch["audio_full_path"][0]
            if config["no_text"] is True:
                prompt_texts = ""
            audio_input, audio_sr = torchaudio.load(audio_full_path)
            audio_input = audio_input.to(device=device, dtype=torch.float16).unsqueeze(0)[:,:,:44100*24]
            waveform = pipeline(
                prompt=prompt_texts,
                audio_end_in_s=2097152/44100,
                guidance_scale=7.0,
                num_inference_steps=50,
                generator=generator,
                initial_audio_waveforms=audio_input,
                initial_audio_sampling_rate=audio_sr
            ).audios
            gen_file_path = os.path.join(output_dir, f"{caption_id[0]}.wav")
            # file_path = os.path.join(output_dir, f"audio_{i}.wav")
            output = waveform[0].T.float().cpu().numpy()
            sf.write(gen_file_path, output, pipeline.vae.sampling_rate)
            dynamics_condition = compute_dynamics(audio_full_path)
            print("dynamics_condition", dynamics_condition.shape)
            gen_dynamics = compute_dynamics(gen_file_path)
            min_len_dynamics = min(gen_dynamics.shape[0], dynamics_condition.shape[0])
            pearson_corr = np.corrcoef(gen_dynamics[:min_len_dynamics], dynamics_condition[:min_len_dynamics])[0, 1]
            print("pearson_corr", pearson_corr)
            score_dynamics.append(pearson_corr)
            melody_condition = extract_melody_one_hot(audio_full_path)     
            print("melody_condition", melody_condition.shape)
            gen_melody = extract_melody_one_hot(gen_file_path)
            min_len_melody = min(gen_melody.shape[1], melody_condition.shape[1])
            matches = ((gen_melody[:, :min_len_melody] == melody_condition[:, :min_len_melody]) & (gen_melody[:, :min_len_melody] == 1)).sum()
            accuracy = matches / min_len_melody
            score_melody.append(accuracy)
            print("melody accuracy", accuracy)
            # Adjust layout to avoid overlap
            processor = RNNDownBeatProcessor()
            input_probabilities = processor(audio_full_path)
            generated_probabilities = processor(gen_file_path)
            hmm_processor = DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)
            input_timestamps = hmm_processor(input_probabilities)
            print("input_probabilities", input_probabilities.shape)
            generated_timestamps = hmm_processor(generated_probabilities)
            precision, recall, f1 = evaluate_f1_rhythm(input_timestamps, generated_timestamps)
            # Output results
            print(f"F1 Score: {f1:.2f}")
            score_rhythm.append(f1)
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
    # result = output[0].float().detach().cpu()
    # torchaudio.save(uri=args.output, src=result, sample_rate=pipeline.vae.sampling_rate)

if __name__ == "__main__":
    main()
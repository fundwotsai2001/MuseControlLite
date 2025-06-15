import os
import json
from utils.extract_conditions import compute_melody, compute_dynamics, compute_dynamics_v2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from madmom.features.downbeats import RNNDownBeatProcessor
import torchaudio
import argparse

# This file is for preparing the musical attribute condition
# Function to process each item
def process_item(idx):
    # try:
    audio_path = os.path.join(data_base_path, meta[idx]['path'])
    # melody_curve = compute_melody(audio_path)
    # rnn_processor = RNNDownBeatProcessor()
    # rhythm_curve = rnn_processor(audio_path)
    dynamics_curve = compute_dynamics_v2(audio_path)
    audio_info = torchaudio.info(audio_path)
    num_frames = audio_info.num_frames
    sample_rate = audio_info.sample_rate
    # print("num_frames", num_frames)
    # print("sample_rate", sample_rate)
    # print("rhythm_curve.shape", rhythm_curve.shape)
    # print("dynamics_curve.shape", dynamics_curve.shape)
    # print("melody_curve.shape", melody_curve.shape)
    assert num_frames == 2097152
    assert sample_rate == 44100
    # assert rhythm_curve.shape == (4756, 2)
    assert dynamics_curve.shape == (2, 13108)
    # assert melody_curve.shape == (128, 4097)
    dynamics_path = os.path.join(dynamics_dir, meta[idx]['path'].replace('.mp3', '.npy').replace("/", "_"))
    # melody_path = os.path.join(melody_dir, meta[idx]['path'].replace('.mp3', '.npy').replace("/", "_"))
    # rhythm_path = os.path.join(rhythm_dir, meta[idx]['path'].replace('.mp3', '.npy').replace("/", "_"))
    np.save(dynamics_path, dynamics_curve)
    # np.save(melody_path, melody_curve)
    # np.save(rhythm_path, rhythm_curve)
    return idx, dynamics_path
    # except Exception as e:
    #     print(f"Error processing {idx}: {e}")
    #     invalid_audio.append(meta[idx]['path'])
    #     return idx, None, None, None

# Multi-processing
if __name__ == "__main__":
    # Paths
    parser = argparse.ArgumentParser(description="Stable-audio VAE encode")
    parser.add_argument("--audio_folder", type=str, default="../mtg_full_47s", help="The audio folder path")
    parser.add_argument("--meta_path", type=str, default="./ALL_condition_wo_SDD_v2.json", help="A list with dictionaries save in a json file")
    parser.add_argument("--new_json", type=str, default="./ALL_condition_wo_SDD_v3.json", help="json file with conditions")
    args = parser.parse_args()  # Parse the arguments

    meta_path = args.meta_path # This is a json file that contains a list of dictionaries, there are two keys in the dictionary: "path" and "Qwen_caption"
    data_base_path = args.audio_folder # audio data root
    new_json = args.new_json # The json file same as meta_path, but added the condition paths
    dynamics_dir = "../mtg_full_47s_conditions/filtered_no_singer_dynamics_test_v2"
    # melody_dir = "./SDD_melody_condition_dir"
    # rhythm_dir = "./SDD_rhythm_condition_dir"

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)
    invalid_audio = []
    os.makedirs(dynamics_dir, exist_ok=True)
    # os.makedirs(melody_dir, exist_ok=True)
    # os.makedirs(rhythm_dir, exist_ok=True)
    num_processes = min(cpu_count(), 20)  # Use up to 8 processes or the number of available CPUs
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_item, range(len(meta))), total=len(meta)))
    # Update metadata
    for idx, dynamics_path, melody_path, rhythm_path in results:
        if dynamics_path:
            meta[idx]['dynamics_path'] = dynamics_path
        # if melody_path:
        #     meta[idx]['melody_path'] = melody_path
        # if rhythm_path:
        #     meta[idx]['rhythm_path'] = rhythm_path
    # Save updated metadata
    with open(new_json, "w") as json_file:
        json.dump(meta, json_file, indent=4)
    print(f"Updated metadata saved to {new_json}")
    print("invalid_audio", invalid_audio)
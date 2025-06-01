import os
import json
from utils.extract_conditions import compute_melody_v2
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import torchaudio
import argparse

def process_item(args):
    idx, meta_item, data_base_path, melody_dir = args
    try:
        audio_path = os.path.join(data_base_path, meta_item['path'])
        melody_curve = compute_melody_v2(audio_path, 44100)
        # print("melody_curve", melody_curve.shape, flush=True)

        audio_info = torchaudio.info(audio_path)
        num_frames = audio_info.num_frames
        sample_rate = audio_info.sample_rate

        assert num_frames == 2097152
        assert sample_rate == 44100
        assert melody_curve.shape == (8, 4097)

        melody_path = os.path.join(melody_dir, meta_item['path'].replace('.mp3', '.npy').replace("/", "_"))
        np.save(melody_path, melody_curve)
        # print(f"Saved {melody_path}")
        return idx, melody_path, None
    except Exception as e:
        print(f"Error processing idx {idx}: {e}")
        return idx, None, meta_item['path']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable-audio VAE encode")
    parser.add_argument("--audio_folder", type=str, default="../mtg_full_47s")
    parser.add_argument("--meta_path", type=str, default="./ALL_condition_wo_SDD.json")
    parser.add_argument("--new_json", type=str, default="./ALL_condition_wo_SDD_v2.json")
    args = parser.parse_args()

    meta_path = args.meta_path
    data_base_path = args.audio_folder
    new_json = args.new_json
    melody_dir = "../mtg_full_47s_conditions/filtered_no_singer_melody_test_new_v3"

    with open(meta_path) as f:
        meta = json.load(f)  # 全部載入

    os.makedirs(melody_dir, exist_ok=True)

    num_processes = min(cpu_count(), 20)
    pool_args = [(i, meta[i], data_base_path, melody_dir) for i in range(len(meta))]

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_item, pool_args), total=len(meta)))

    invalid_audio = []
    for idx, melody_path, invalid_path in results:
        if melody_path:
            meta[idx]['melody_path'] = melody_path
        if invalid_path:
            invalid_audio.append(invalid_path)

    with open(new_json, "w") as json_file:
        json.dump(meta, json_file, indent=4)

    print(f"Updated metadata saved to {new_json}")
    print("Invalid audio files:", invalid_audio)

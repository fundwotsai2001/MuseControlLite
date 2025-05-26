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
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            # if i >= config["test_num"]:
            #     break
            caption_id = batch["caption_id"]
            prompt_texts = batch["prompt_texts"]
            audio_full_path = batch["audio_full_path"]
            if config["no_text"] is True:
                prompt_texts = ""
            audio_input, audio_sr = torchaudio.load(audio_full_path[0])
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
            file_path = os.path.join(output_dir, f"{caption_id[0]}.wav")
            # file_path = os.path.join(output_dir, f"audio_{i}.wav")
            output = waveform[0].T.float().cpu().numpy()
            sf.write(file_path, output, pipeline.vae.sampling_rate)
            with open(file_path, "rb") as f:
                os.fsync(f.fileno())
    # result = output[0].float().detach().cpu()
    # torchaudio.save(uri=args.output, src=result, sample_rate=pipeline.vae.sampling_rate)

if __name__ == "__main__":
    main()
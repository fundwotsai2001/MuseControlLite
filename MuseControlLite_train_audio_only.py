import argparse
import itertools
import math
import os
import random
import shutil
import warnings
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import Dataset, random_split, DataLoader
import torchaudio
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from scipy.io.wavfile import write
from scipy.signal import savgol_filter
from diffusers.loaders import AttnProcsLayers
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file  # Import safetensors
warnings.filterwarnings("ignore", category=FutureWarning)
from utils.stable_audio_dataset_utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
from torchaudio import transforms as T
import torch
import soundfile as sf
from pipeline.stable_audio_multi_cfg_pipe import StableAudioPipeline
from diffusers.loaders import AttnProcsLayers
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
    StableAudioAttnProcessor2_0_rotary_double,
)
from utils.extract_conditions import compute_dynamics, extract_melody_one_hot, evaluate_f1_rhythm
from config_training import get_config
import gc
torchaudio.set_audio_backend("sox_io")
import torch.distributed as dist

class AudioInversionDataset(Dataset):
    def __init__(
        self,
        config,
        audio_codec_root,
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
        self.audio_codec_root = audio_codec_root
        self.device = device
        self.meta_path = config['meta_data_path']
        with open(self.meta_path) as f:
            self.meta = json.load(f)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):    
        meta_entry = self.meta[i]
        audio_path = meta_entry.get('path')
        
        # Load audio tokens, they are encoded with the Stable-audio VAE and saved, skipping the the VAE encoding process saves memory when training MuseControlLite
        audio_full_path = os.path.join(self.audio_data_root, audio_path)
        audio_token_path = os.path.join(self.audio_codec_root, audio_path.replace('mp3', 'pth'))
        audio = torch.load(audio_token_path, map_location=torch.device('cpu'))
        
        example = {
            "text": meta_entry['Qwen_caption'],
            "audio_full_path": audio_full_path,
            "audio": audio,
            "seconds_start": 0,
            "seconds_end": 2097152 / 44100,
        }
        return example
    
class CollateFunction:
    def __init__(self, condition_type):
        self.condition_type = condition_type
    def __call__(self, examples):
        audio = [example["audio"] for example in examples]
        prompt_texts = [example["text"] for example in examples]
        audio_full_path = [example["audio_full_path"] for example in examples]
        seconds_start = [example["seconds_start"] for example in examples]
        seconds_end = [example["seconds_end"] for example in examples]
        audio = torch.stack(audio)
        batch = {
            "audio_full_path": audio_full_path,
            "audio": audio,
            "prompt_texts": prompt_texts,
            "seconds_start": seconds_start,
            "seconds_end": seconds_end,
        }

        return batch

def log_validation(val_dataloader, condition_type, pipeline, config, weight_dtype, global_step):
    val_audio_dir = os.path.join(config["output_dir"], "val_audio_{}".format(global_step))
    os.makedirs(val_audio_dir, exist_ok=True)
    score_dynamics = []
    score_melody = []
    score_rhythm = []
    for step, batch in enumerate(val_dataloader):
        if step > config["test_num"]:
            break
        pipeline.transformer.eval()  # Set the transformer to evaluation mode
        prompt_texts = batch["prompt_texts"]
        audio_full_path = batch["audio_full_path"]
        ### conditioned
        audio_condition = batch["audio"]
        desired_repeats = 768 // 64  # Number of repeats needed
        extracted_audio_condition = audio_condition.repeat_interleave(desired_repeats, dim=1)
        masked_extracted_audio_condition = torch.full_like(extracted_audio_condition.to(torch.float32), fill_value=0)
        if step < 3:
            extracted_audio_condition[:,:,512:] = 0
        elif step < 6:
            extracted_audio_condition[:,:,:512] = 0
        extracted_condition = torch.concat((masked_extracted_audio_condition, masked_extracted_audio_condition, extracted_audio_condition), dim=0)
        extracted_condition = extracted_condition.transpose(1, 2)
        generator = torch.Generator("cuda").manual_seed(0)
        with torch.no_grad():
            audio = pipeline(
                extracted_condition = extracted_condition, 
                guidance_scale_con = config['guidance_scale_con'],
                guidance_scale_text=config["guidance_scale_text"],
                prompt=prompt_texts,
                negative_prompt=[""],
                num_inference_steps=config["denoise_step"],
                audio_end_in_s=2097152/44100,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios
        output = audio[0].T.float().cpu().numpy()
        gen_file = os.path.join(val_audio_dir, f"validation_{step}.wav")
        original_file = os.path.join(val_audio_dir, f"original_{step}.wav")
        sf.write(gen_file, output, pipeline.vae.sampling_rate)
        shutil.copy(audio_full_path[0], original_file)
        discription_path = os.path.join(val_audio_dir, "description.txt")
        with open(discription_path, 'a') as file:
            file.write(f'{prompt_texts}\n')
    torch.cuda.empty_cache()
    gc.collect()
    return np.mean(score_dynamics), np.mean(score_melody), np.mean(score_rhythm)
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
def check_and_print_non_float32_parameters(model):
    non_float32_params = []
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            non_float32_params.append((name, param.dtype))
    
    if non_float32_params:
        print("Not all parameters are in float32!")
        print("The following parameters are not in float32:")
        for name, dtype in non_float32_params:
            print(f"Parameter: {name}, Data Type: {dtype}")
    else:
        print("All parameters are in float32.")

def main():
    torch.manual_seed(42)
    config = get_config()
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
        log_with="wandb",
    )

    if not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if config["output_dir"] is not None:
            os.makedirs(config["output_dir"], exist_ok=True)
    # decide weight precision for freezed models
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # initialize models
    pipeline = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype).to("cuda")
    text_encoder=pipeline.text_encoder
    projection_model=pipeline.projection_model
    vae=pipeline.vae
    noise_scheduler=pipeline.scheduler
    noise_scheduler.config.sigma_max = config["sigma_max"]
    noise_scheduler.config.sigma_min = config["sigma_min"]
    transformer = pipeline.transformer

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    projection_model.requires_grad_(False)

    # Define a dictionary to map types to corresponding processor classes, currently only "rotary" is available.
    processor_classes = {
        "rotary": StableAudioAttnProcessor2_0_rotary,
        "rotary_double": StableAudioAttnProcessor2_0_rotary_double,
    }
    print(config["attn_processor_type"])
    # Get the processor classes based on the type
    attn_processor = processor_classes.get(config["attn_processor_type"], None)
    attn_procs = {}
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
            ).to("cuda", dtype=torch.float32)
    # Load checkpoint
    if config["transformer_ckpt"] is not None:
        if "bin" in config["transformer_ckpt"]:
            state_dict = torch.load(config["transformer_ckpt"])
        elif "safetensors" in config["transformer_ckpt"]:
            state_dict = load_file(config["transformer_ckpt"], device="cpu")
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
            return pipeline.transformer(*args, **kwargs)

    transformer = _Wrapper(pipeline.transformer.attn_processors)
    optimizer_class = torch.optim.AdamW
    params_to_optimize = itertools.chain(
        transformer.parameters(),
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay= config['weight_decay'],
        eps=1e-08,
    )

    # Dataset and DataLoaders creation:
    dataset = AudioInversionDataset(
        config,
        audio_codec_root=config['audio_codec_root'],
        audio_data_root=config["audio_data_dir"],
        device=accelerator.device,
        )
    val_size =  config["validation_num"]
    train_size = len(dataset) - val_size 

    # Ensure consistent splitting
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # DataLoader
    train_collate_fn = CollateFunction(condition_type=config["condition_type"])
    val_collate_fn = CollateFunction(condition_type=config["condition_type"])
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=val_collate_fn,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
    if config["max_train_steps"] is None:
        config["max_train_steps"] = config["num_train_epochs"] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config['lr_scheduler'],
        optimizer=optimizer,
        step_rules = None,
        num_warmup_steps = 500,
        num_training_steps = config['max_train_steps'],
        num_cycles = 1,
        power = 1.0,
        last_epoch = -1,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
    if overrode_max_train_steps:
        config["max_train_steps"] = config["num_train_epochs"] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config["num_train_epochs"] = math.ceil(config["max_train_steps"] / num_update_steps_per_epoch)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="MuseControlLite",      # your W&B project
            config=config,                        # whatever hyperparams you’re logging
            init_kwargs={
                "wandb": {
                    "name": config['wand_run_name'],   # <— your chosen run name
                }
            }
        )
    global_step = 0
    first_epoch = 0
    score_melody = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config["max_train_steps"]), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    print("log_validation_first", config["log_first"])
    score_dynamics, score_melody, score_rhythm = 0, 0, 0
    if config["log_first"] and accelerator.is_main_process:
        score_dynamics, score_melody, score_rhythm = log_validation(val_dataloader,
                            config["condition_type"],
                            pipeline, config, weight_dtype, global_step
                        )
    for epoch in range(first_epoch, config["num_train_epochs"]):
        for step, batch in enumerate(train_dataloader):
            transformer.train()
            with accelerator.accumulate(transformer):
                # Convert audios to latent space
                latents = batch["audio"]
                bsz, channels, height = latents.shape
                # Sample a random timestep for each image using uniform distribution
                t = torch.sigmoid(torch.randn(bsz)).cuda().half()
                # Calculate the noise schedule parameters for those timesteps
                alphas, sigmas = get_alphas_sigmas(t)  # get_alphas_sigmas should be defined as in the wrapper
                alphas = alphas[:, None, None]  # Shape to match latents
                sigmas = sigmas[:, None, None]
                # Sample noise and add it to the latents
                noise = torch.randn_like(latents)
                noisy_latents = latents * alphas + noise * sigmas
                # Determine the target for v_prediction
                if noise_scheduler.config.prediction_type == "v_prediction":
                    targets = alphas * noise - sigmas * latents
                else:
                    targets = noise  # For epsilon, the target is just the noise
                prompt_texts = batch["prompt_texts"]
                desired_repeats = 768 // 64  # Number of repeats needed
                extracted_audio_condition = latents.repeat_interleave(desired_repeats, dim=1)
                for i in range(len(prompt_texts)):
                    rand_num = random.random()
                    num1, num2 = random.sample(range(1024), 2)
                    # 50% chance to set prompt_texts[i] to an empty string
                    if rand_num < 0.1:
                        ## all blank
                        prompt_texts[i] = ""
                        extracted_audio_condition[i] = torch.zeros_like(extracted_audio_condition[i])                  
                    elif rand_num < 0.4:
                        ## 0~num1 : melody, rhythm, dynamics or blank
                        ## num1~num2 : audio or blank
                        ## num2~1024: melody, rhythm, dynamics or blank
                        if random.random() < 0.3:
                            prompt_texts[i] = ""
                        if random.random() < 0.7:
                            extracted_audio_condition[i][:,  : num1] = 0
                            extracted_audio_condition[i][:, num2 : ] = 0
                        else:
                            extracted_audio_condition[i][:, : ] = 0
                    elif rand_num < 0.7:
                        ## 0~num1 : audio or blank
                        ## num1~num2 : melody, rhythm, dynamics or blank
                        ## num2~1024: audio or blank
                        if random.random() < 0.3:
                            prompt_texts[i] = ""
                        if random.random() < 0.7:
                            extracted_audio_condition[i][:, num1: num2] = 0
                        else:
                            extracted_audio_condition[i][:, : ] = 0
                if "audio" not in config['condition_type']:
                    extracted_audio_condition[:,:,:] = 0
                    print("not using auio")
                with torch.no_grad():
                    prompt_embeds = pipeline.encode_prompt(
                        prompt=prompt_texts,
                        device="cuda",
                        do_classifier_free_guidance=False,
                    )
                    batch_size = len(prompt_texts)
                    audio_start_in_s = batch["seconds_start"]
                    audio_end_in_s = batch["seconds_end"]
                    # Encode duration
                    seconds_start_hidden_states, seconds_end_hidden_states = pipeline.encode_duration(
                        audio_start_in_s,
                        audio_end_in_s,
                        device="cuda",
                        do_classifier_free_guidance=False,
                        batch_size=batch_size,
                    )
                
                audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2)
                text_audio_duration_embeds = torch.cat(
                    [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
                ) 
                extracted_condition = extracted_audio_condition.transpose(1, 2)
                # This rotary_embedding is for self attention layers in Stable-audio 
                rotary_embed_dim = pipeline.transformer.config.attention_head_dim // 2
                rotary_embedding = get_1d_rotary_pos_embed(
                    rotary_embed_dim,
                    latents.shape[2] + audio_duration_embeds.shape[1],
                    use_real=True,
                    repeat_interleave_real=False,
                )              
                with accelerator.autocast():
                    model_pred = pipeline.transformer(
                        noisy_latents,
                        t,  # Use continuous t for conditioning
                        encoder_hidden_states=text_audio_duration_embeds,
                        encoder_hidden_states_con = extracted_condition,
                        global_hidden_states=audio_duration_embeds,
                        rotary_embedding=rotary_embedding,
                        return_dict=False,
                    )[0]
                    # Compute the loss
                    loss = F.mse_loss(model_pred, targets, reduction="mean")
                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(
                            transformer.parameters(),
                        )
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                audios = []
                progress_bar.update(1)
                global_step += 1
            
                if accelerator.is_main_process:
                    if global_step % config["checkpointing_steps"] == 0:
                        save_path = os.path.join(config["output_dir"], f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # logger.info(f"Saved state to {save_path}")

                    if global_step % config["validation_steps"] == 0:
                        score_dynamics, score_melody, score_rhythm = log_validation(val_dataloader,
                            config["condition_type"],
                            pipeline, config, weight_dtype, global_step
                        )
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config["max_train_steps"]:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
    

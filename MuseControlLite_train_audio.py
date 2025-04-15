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
import torch.nn as nn
import torch.utils.checkpoint
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
from stable_audio_dataset_utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
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
from SDD_melody_acc import extract_melody_one_hot
from sklearn.metrics import f1_score
from config_stable_audio_training import get_config
from torch.cuda.amp import autocast
from utils.dynamics_control_for_stable_audio import compute_dynamics
from utils.melody_extract import compute_melody
from madmom.features import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor,RNNDownBeatProcessor
### same way as stable audio loads audio file
import gc
torchaudio.set_audio_backend("sox_io")
import datetime
import torch.distributed as dist
import time



def print_tensor_info(tensor, tensor_name):
    print(f"{tensor_name} - Device: {tensor.device}, Shape: {tensor.shape}, Version: {tensor._version}")
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def load_audio_file(filename):
    try:
        # Use torchaudio to load the file regardless of format
        audio, in_sr = torchaudio.load(filename)
        
        # Resample if necessary
        if in_sr != 44100:
            resample_tf = T.Resample(in_sr, 44100)
            audio = resample_tf(audio)
        
        return audio

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except RuntimeError as e:
        print(f"Error: Unable to process file {filename}. {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while loading {filename}: {e}")
        return None
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
    def __init__(self, condition_type, mode="train"):
        self.condition_type = condition_type
        self.mode = mode  # "train" or "val"
    def __call__(self, examples):
        audio = [example["audio"] for example in examples]
        prompt_texts = [example["text"] for example in examples]
        audio_full_path = [example["audio_full_path"] for example in examples]
        seconds_start = [example["seconds_start"] for example in examples]
        seconds_end = [example["seconds_end"] for example in examples]
        if len(self.condition_type) != 0:
            audio = torch.stack(audio).float()   
            batch = {
                "audio_full_path": audio_full_path,
                "audio": audio,
                "prompt_texts": prompt_texts,
                "seconds_start": seconds_start,
                "seconds_end": seconds_end,
            }
        else:
            audio = torch.stack(audio).to(memory_format=torch.contiguous_format).float()
            batch = {
                # "audio_full_path": audio_full_path,
                "audio": audio,
                "prompt_texts": prompt_texts,
                "seconds_start": seconds_start,
                "seconds_end": seconds_end,
            }

        return batch
class melody_extractor(nn.Module):
    def __init__(self):
        super(melody_extractor, self).__init__()
        self.conv1d_1 = nn.Conv1d(128, 128, kernel_size=3, padding=0, stride=2)  
        self.conv1d_2 = nn.Conv1d(128, 192, kernel_size=3, padding=1, stride=2)  
        self.conv1d_3 = nn.Conv1d(192, 192, kernel_size=3, padding=1)
    def forward(self, x):
        # original shape: (batchsize, 12, 1296)
        x = self.conv1d_1(x)# shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 64, 2048)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 128, 1024)
        return x
class dynamics_extractor(nn.Module):
    def __init__(self):
        super(dynamics_extractor, self).__init__()
        self.conv1d_1 = nn.Conv1d(1, 16, kernel_size=3, padding=1, stride=2)  
        self.conv1d_2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(16, 128, kernel_size=3, padding=1, stride=2)
        self.conv1d_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(128, 192, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        # original shape: (batchsize, 1, 8280)
        # x = x.unsqueeze(1) # shape: (batchsize, 1, 8280)
        x = self.conv1d_1(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_2(x)  # shape: (batchsize, 16, 4140)
        x = F.silu(x)
        x = self.conv1d_3(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_4(x)  # shape: (batchsize, 128, 2070)
        x = F.silu(x)
        x = self.conv1d_5(x)  # shape: (batchsize, 192, 1035)
        return x
class rhythm_extractor(nn.Module):
    def __init__(self):
        super(rhythm_extractor, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, 16, kernel_size=3, padding=1)  
        self.conv1d_2 = nn.Conv1d(16, 64, kernel_size=3, padding=1)  
        self.conv1d_3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2)  
        self.conv1d_4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(128, 192, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        # original shape: (batchsize, 2, 3000)
        x = self.conv1d_1(x)# shape: (batchsize, 64, 3000)
        x = F.silu(x)
        x = self.conv1d_2(x) # shape: (batchsize, 64, 3000)
        x = F.silu(x)
        x = self.conv1d_3(x) # shape: (batchsize, 128, 1500)
        x = F.silu(x)
        x = self.conv1d_4(x) # shape: (batchsize, 128, 1500)
        x = F.silu(x)
        x = self.conv1d_5(x) # shape: (batchsize, 192, 750)
        return x
import numpy as np

def evaluate_f1_rhythm(input_timestamps, generated_timestamps, tolerance=0.07):
    """
    Evaluates precision, recall, and F1-score for beat/downbeat timestamp alignment.
    
    Args:
        input_timestamps (ndarray): 2D array of shape [n, 2], where column 0 contains timestamps.
        generated_timestamps (ndarray): 2D array of shape [m, 2], where column 0 contains timestamps.
        tolerance (float): Alignment tolerance in seconds (default: 70ms).
    
    Returns:
        tuple: (precision, recall, f1)
    """
    # Extract and sort timestamps
    input_timestamps = np.asarray(input_timestamps)
    generated_timestamps = np.asarray(generated_timestamps)
    
    # If you only need the first column
    if input_timestamps.size > 0:  
        input_timestamps = input_timestamps[:, 0]
        input_timestamps.sort()
    else:
        input_timestamps = np.array([])
        
    if generated_timestamps.size > 0:
        generated_timestamps = generated_timestamps[:, 0]
        generated_timestamps.sort()
    else:
        generated_timestamps = np.array([])

    # Handle empty cases
    # Case 1: Both are empty
    if len(input_timestamps) == 0 and len(generated_timestamps) == 0:
        # You could argue everything is correct since there's nothing to detect,
        # but returning all zeros is a common convention.
        return 0.0, 0.0, 0.0

    # Case 2: No ground-truth timestamps, but predictions exist
    if len(input_timestamps) == 0 and len(generated_timestamps) > 0:
        # All predictions are false positives => tp=0, fp = len(generated_timestamps)
        # => precision=0, recall is undefined (tp+fn=0), typically we treat recall=0
        return 0.0, 0.0, 0.0

    # Case 3: Ground-truth timestamps exist, but no predictions
    if len(input_timestamps) > 0 and len(generated_timestamps) == 0:
        # Everything in input_timestamps is a false negative => tp=0, fn = len(input_timestamps)
        # => recall=0, precision is undefined (tp+fp=0), typically we treat precision=0
        return 0.0, 0.0, 0.0

    # If we get here, both arrays are non-empty
    tp = 0
    fp = 0
    
    # Track matched ground-truth timestamps
    matched_inputs = np.zeros(len(input_timestamps), dtype=bool)
    
    for gen_ts in generated_timestamps:
        # Calculate absolute differences to each reference timestamp
        diffs = np.abs(input_timestamps - gen_ts)
        # Find index of the closest input timestamp
        min_diff_idx = np.argmin(diffs)
        
        # Check if that difference is within tolerance and unmatched
        if diffs[min_diff_idx] < tolerance and not matched_inputs[min_diff_idx]:
            tp += 1
            matched_inputs[min_diff_idx] = True
        else:
            fp += 1  # no suitable match found or closest was already matched
    
    # Remaining unmatched input timestamps are false negatives
    fn = np.sum(~matched_inputs)
    
    # Compute precision, recall, f1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def log_validation(val_dataloader, condition_extractors, condition_type, pipeline, config, weight_dtype, global_step):
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
        extracted_audio_condition[:,:,step*150:(step+1)*150] = 0 # pause audio condition
        extracted_condition = torch.concat((masked_extracted_audio_condition, masked_extracted_audio_condition, extracted_audio_condition), dim=0)
        extracted_condition = extracted_condition.transpose(1, 2)
        generator = torch.Generator("cuda").manual_seed(0)
        # print("extracted_condition", extracted_condition.shape)
        # print("test cfg!!")
        with torch.no_grad():
            audio = pipeline(
                extracted_condition = extracted_condition, 
                guidance_scale_con = 1.0,
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
        # original_audio = batch["audio"][0].T.float().cpu().numpy()
        sf.write(gen_file, output, pipeline.vae.sampling_rate)
        shutil.copy(audio_full_path[0], original_file)
        discription_path = os.path.join(val_audio_dir, "description.txt")
        with open(discription_path, 'a') as file:
            file.write(f'{prompt_texts}\n')
    gc.collect()
    return 0, 0, 0
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
    # decide weight precision
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

    # # initialize condition extractors
    condition_extractors = {}
    if config["apadapter"]:
        melody_conditoner = melody_extractor().cuda().float()
        condition_extractors["melody"] = melody_conditoner
        dynamics_conditoner = dynamics_extractor().cuda().float()
        condition_extractors["dynamics"] = dynamics_conditoner
        rhythm_conditoner = rhythm_extractor().cuda().float()
        condition_extractors["rhythm"] = rhythm_conditoner
        for conditioner in condition_extractors.values():
            conditioner.requires_grad_(True)

    # load pretrained condition extractors
    for conditioner_type, ckpt_path in config["extractor_ckpt"].items():
        if "bin" in ckpt_path:
            state_dict = torch.load(ckpt_path)
        elif "safetensors" in ckpt_path:
            state_dict = load_file(ckpt_path, device="cpu")
        condition_extractors[conditioner_type].load_state_dict(state_dict)
        print(f"load checkpoint from {config['extractor_ckpt']} successfully !")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    projection_model.requires_grad_(False)

    # Define a dictionary to map types to corresponding processor classes
    processor_classes = {
        "rotary": StableAudioAttnProcessor2_0_rotary,
        "rotary_double": StableAudioAttnProcessor2_0_rotary_double,
    }
    print(config["attn_processor_type"])
    # Get the processor classes based on the type
    attn_processor = processor_classes.get(config["attn_processor_type"], None)
    if config["apadapter"]:
        print("use apadapter")
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
                    scale=1.0,
                ).to("cuda", dtype=torch.float32)

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
                    param_sum = processor.conv_out.weight.sum()
                    # print("param_sum", param_sum)
                    print(f"load {name}")
        transformer.set_attn_processor(attn_procs)
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.transformer(*args, **kwargs)

        transformer = _Wrapper(pipeline.transformer.attn_processors)
    else:
        transformer = pipeline.transformer



    optimizer_class = torch.optim.AdamW
    if config["apadapter"]:
        params_to_optimize = itertools.chain(
            transformer.parameters(),
            *[model.parameters() for model in condition_extractors.values()]
        )
    else:
        params_to_optimize = (
            itertools.chain(transformer.parameters())
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
    train_size = len(dataset) - val_size  # Remaining 20% for validation

    # Ensure consistent splitting
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # DataLoader
    train_collate_fn = CollateFunction(condition_type=config["condition_type"], mode="train")
    val_collate_fn = CollateFunction(condition_type=config["condition_type"], mode="val")
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
    print(len(val_dataloader))

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
    if config["max_train_steps"] is None:
        config["max_train_steps"] = config["num_train_epochs"] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
    )

    # Prepare everything with our `accelerator`.
    if config["apadapter"]:
        condition_extractor_values = list(condition_extractors.values())

        *condition_extractor_values, transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            *condition_extractor_values, transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

    else:
        transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
    if overrode_max_train_steps:
        config["max_train_steps"] = config["num_train_epochs"] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config["num_train_epochs"] = math.ceil(config["max_train_steps"] / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth_audio", config)

    global_step = 0
    first_epoch = 0
    score_melody = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config["max_train_steps"]), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    print("log_first", config["log_first"])
    score_dynamics, score_melody, score_rhythm = 0, 0, 0
    if config["log_first"] and accelerator.is_main_process:
        score_dynamics, score_melody, score_rhythm = log_validation(val_dataloader,
                            condition_extractors,
                            config["condition_type"],
                            pipeline, config, weight_dtype, global_step
                        )
    for epoch in range(first_epoch, config["num_train_epochs"]):
        for step, batch in enumerate(train_dataloader):
            transformer.train()
            if config["apadapter"]:
                for model in condition_extractors.values():
                    model.train()
            else:
                condition_extractor_values = None
            with accelerator.accumulate(transformer, *condition_extractor_values if condition_extractor_values is not None else transformer):
                # Convert audios to latent space
                latents = batch["audio"]
                bsz, channels, height = latents.shape
                # Sample a random timestep for each image using uniform distribution
                t = torch.sigmoid(torch.randn(bsz)).cuda()
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
                extracted_audio_condition = batch["audio"].repeat_interleave(desired_repeats, dim=1)
                for i in range(len(prompt_texts)):
                    rand_num = random.random()
                    num1, num2 = random.sample(range(1024), 2)
                    # 50% chance to set prompt_texts[i] to an empty string
                    if rand_num < 0.1:
                        prompt_texts[i] = ""
                    # 10% chance to zero out *all* conditions                
                    elif rand_num < 0.55:
                        ## 0~num1 : melody, rhythm, dynamics or blank
                        ## num1~num2 : audio or blank
                        ## num2~1024: melody, rhythm, dynamics or blank
                        if random.random() < 0.2:
                            prompt_texts[i] = ""
                        if random.random() < 0.2:
                            extracted_audio_condition[i][:,  : num1] = 0
                            extracted_audio_condition[i][:, num2 : ] = 0
                        else:
                            extracted_audio_condition[i][:, : ] = 0
                    else:
                        ## 0~num1 : audio or blank
                        ## num1~num2 : melody, rhythm, dynamics or blank
                        ## num2~1024: audio or blank
                        if random.random() < 0.2:
                            prompt_texts[i] = ""
                        if random.random() < 0.2:
                            extracted_audio_condition[i][:, num1: num2] = 0
                        else:
                            extracted_audio_condition[i][:, : ] = 0
    
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
                
                audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2).float()
                text_audio_duration_embeds = torch.cat(
                    [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
                )
                num_waveforms_per_prompt = 1
                # print("extracted_condition", extracted_condition.shape)               
                extracted_audio_condition = extracted_audio_condition.transpose(1, 2)
                bs_embed, seq_len, hidden_size = text_audio_duration_embeds.shape
                # Duplicate audio_duration_embeds and text_audio_duration_embeds
                text_audio_duration_embeds = text_audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
                text_audio_duration_embeds = text_audio_duration_embeds.view(
                    bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
                )

                audio_duration_embeds = audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
                audio_duration_embeds = audio_duration_embeds.view(
                    bs_embed * num_waveforms_per_prompt, -1, audio_duration_embeds.shape[-1]
                )

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
                        encoder_hidden_states_con = extracted_audio_condition,
                        global_hidden_states=audio_duration_embeds,
                        rotary_embedding=rotary_embedding,
                        return_dict=False,
                    )[0]
                    # Compute the loss
                    loss = F.mse_loss(model_pred.float(), targets.float(), reduction="mean")
                # Backpropagation
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # print("clip here")
                    if config["apadapter"]:
                        params_to_clip = (
                            itertools.chain(
                                transformer.parameters(),
                                *[model.parameters() for model in condition_extractors.values()]
                            )
                        )
                    else:
                        params_to_clip = (
                            itertools.chain(transformer.parameters())
                        )
                    # for name, param in transformer.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"Transformer Gradient: {name} -> {param.grad.dtype}")

                    # if config["apadapter"]:
                    #     for key, model in condition_extractors.items():
                    #         for name, param in model.named_parameters():
                    #             if param.grad is not None:
                    #                 print(f"Condition Extractor Gradient [{key}]: {name} -> {param.grad.dtype}")

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
                            condition_extractors,
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
    

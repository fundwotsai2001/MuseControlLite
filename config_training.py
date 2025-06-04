def get_config():
    return {
        # Load files and checkpoints

        "condition_type": ["melody", "rhythm", "dynamics"], #"melody", "rhythm", "dynamics", "audio"

        "meta_data_path": "./ALL_condition_wo_SDD_v2.json",

        "audio_data_dir": "../mtg_full_47s",

        "audio_codec_root": "../mtg_full_47s_codec",

        "output_dir": "./checkpoints/ALL_new_melody",

        "transformer_ckpt": None, #"./checkpoints/stable_audio_new_fast_audio_v2/checkpoint-21000/model.safetensors",

        "extractor_ckpt": {
            # "dynamics": "./checkpoints/110000_musical_44000_audio/model_1.safetensors",
            # "melody": "./checkpoints/stable_audio_melody_wo_SDD/checkpoint-5000/model.safetensors",
            # "rhythm": "./checkpoints/110000_musical_44000_audio/model_2.safetensors",
        },

        "wand_run_name": "ALL_new_melody",

        # training hyperparameters
        "GPU_id" : "2",

        "train_batch_size": 16,

        "learning_rate": 1e-4,

        "attn_processor_type": "rotary", # "rotary", "rotary_conv_in", "absolute" 

        "gradient_accumulation_steps": 8,

        "max_train_steps": 200000,

        "num_train_epochs": 20,

        "dataloader_num_workers": 16,

        "mixed_precision": "fp16", #["no", "fp16", "bf16"]

        "apadapter": True,

        "lr_scheduler": "linear", # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'

        "weight_decay": 1e-2,

        #config for validation
        "validation_num": 1000,

        "test_num": 5,

        "ap_scale": 1.0,

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 1.0, # The separated guidance for both Musical attribute and audio conditions. Note that if guidance scale is too large, the audio quality will be bad. Values between 0.5~2.0 is recommended.

        "checkpointing_steps": 500,

        "validation_steps": 500,

        "denoise_step": 50,

        "log_first": True,

        "sigma_min": 0.3,

        "sigma_max": 500,
    }
def get_config():
    return {
        # Load files and checkpoints
        "condition_type": ["melody", "rhythm", "dynamics", "audio"], #"melody", "rhythm", "dynamics", "audio"
        "meta_data_path": "./Qwen_caption.json",
        "audio_data_dir": "../mtg_full_47s",
        "audio_codec_root": "../mtg_full_47s_codec",
        "output_dir": "./checkpoints/stable_audio_new_fast_all_v3",
        "transformer_ckpt": "./checkpoints/110000_musical_attribute_checkpoint/model_3.safetensors",
        "extractor_ckpt": {
            "dynamics": "./checkpoints/110000_musical_attribute_checkpoint/model_1.safetensors",
            "melody": "./checkpoints/110000_musical_attribute_checkpoint/model.safetensors",
            "rhythm": "./checkpoints/110000_musical_attribute_checkpoint/model_2.safetensors",
        },
        # training hyperparameters
        "GPU_id" : "0",
        "train_batch_size": 4,
        "learning_rate": 1e-4,
        "attn_processor_type": "rotary", # "rotary", "rotary_conv_in", "absolute" 
        "gradient_accumulation_steps": 8,
        "max_train_steps": 1000000,
        "num_train_epochs": 20,
        "dataloader_num_workers": 16,
        "mixed_precision": "fp16", #["no", "fp16", "bf16"]
        "apadapter": True,
        "lr_scheduler": "constant", # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
        "weight_decay": 1e-4,

        #config for validation
        "validation_num": 1000,
        "test_num": 5,
        "ap_scale": 1.0,
        "guidance_scale": 7.0,   
        "checkpointing_steps": 1000,
        "validation_steps": 500,
        "denoise_step": 50,
        "log_first": False,
        "sigma_min": 0.3,
        "sigma_max": 500,
        "audio_pooling_rate": [1, 2, 4, 8]
    }
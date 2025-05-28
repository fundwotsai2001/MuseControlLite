def get_config():
    return {
        "condition_type": ["melody"], # options: "dynamics", "rhythm", "melody", "audio". You can choose any combinations you like

        "output_dir": "./generated_audio/melody_only",

        # Works for all combinations
        ###############
        "transformer_ckpt": "./checkpoints/woSDD-all/model_3.safetensors",
        
        "extractor_ckpt": {
            "dynamics": "./checkpoints/woSDD-all/model_1.safetensors",
            "melody": "./checkpoints/woSDD-all/model.safetensors",
            "rhythm": "./checkpoints/woSDD-all/model_2.safetensors",
        },
        ###############

        # When using audio only, do not need extractor
        #################
        "audio_transformer_ckpt": "./checkpoints/Audio_only-39500/model.safetensors",

        # For melody only.
        ###############
        "transformer_ckpt_melody": "./checkpoints/40000_Melody_only/model_1.safetensors",

        "extractor_ckpt_melody": {
            "melody": "./checkpoints/40000_Melody_only/model.safetensors",
        },
        ###############

        "GPU_id": "2",

        "attn_processor_type": "rotary", 

        "apadapter": True, # True for MuseControlLite, False for original Stable-audio

        "ap_scale": 1.0, # recommend 1.0 for MuseControlLite, other values are not tested

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 2.0, # The separated guidance for both Musical attribute and audio conditions. Note that if guidance scale is too large, the audio quality will be bad. Values between 0.5~2.0 is recommended.
        
        "denoise_step": 100,

        "sigma_min": 0.3, # sigma_min and sigma_max are for the scheduler.

        "sigma_max": 500,  # Note that if sigma_max is too large or too small, the "audio condition generation" will be bad.

        "weight_dtype": "fp32", # fp16 and fp32 sounds quiet the same.

        "negative_text_prompt": "Low qualiy, noise",

        # The below two mask should complementary, which means every time slice shouldn't receive both audio and music attribute condition.
        # Don't set both use_audio_mask and use_musical_attribute_mask to True.

        ###############
        "use_audio_mask": False,

        "audio_mask_start_seconds": 24,

        "audio_mask_end_seconds": 2097152 / 44100, # Maximum duration for stable-audio is 2097152 / 44100 seconds

        "use_musical_attribute_mask": False, # Set to true, if you want to mask melody, rhythm, dynamics.

        "musical_attribute_mask_start_seconds": 24,

        "musical_attribute_mask_end_seconds": 2097152 / 44100 ,
        ###############

        "no_text": False, # Optional, set to true if no text prompt is needed (possible for audio inpainting or outpainting)

        "audio_files": [
            "./melody_condition_audio/49_piano.mp3",
            "./melody_condition_audio/322_piano.mp3",
            "./melody_condition_audio/610_bass.mp3",
            "./melody_condition_audio/785_piano.mp3",
            "./melody_condition_audio/933_string.mp3",
            "./melody_condition_audio/57_jazz.mp3",
            "./melody_condition_audio/703_mideast.mp3"
        ],

        "text": [
                # "a recording of a piano",
                # "a recording of a piano",
                "A heartfelt, warm acoustic guitar performance, evoking a sense of tenderness and deep emotion, with a melody that truly resonates and touches the heart.",     
                "A vibrant MIDI electronic composition with a hopeful and optimistic vibe.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "Hopeful instrumental with guitar being the lead and tabla used for percussion in the middle giving a feeling of going somewhere with positive outlook.",
                "A string ensemble opens the track with legato, melancholic melodies. The violins and violas play beautifully, while the cellos and bass provide harmonic support for the moving passages. The overall feel is deeply melancholic, with an emotionally stirring performance that remains harmonious and a sense of clearness.",
                "An exceptionally harmonious string performance with a lively tempo in the first half, transitioning to a gentle and beautiful melody in the second half. It creates a warm and comforting atmosphere, featuring cellos and bass providing a solid foundation, while violins and violas showcase the main theme, all without any noise, resulting in a cohesive and serene sound.",
                ]
    }
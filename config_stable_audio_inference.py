def get_config():
    return {
        "condition_type": ["melody"], # options: "dynamics", "rhythm", "melody", "audio"
        "output_dir": "./generated_audio/melody_audio",
        #Checkpoints
        ###############
        "transformer_ckpt": "./checkpoints/stable_audio_new_fast_all_v2/checkpoint-44000/model_3.safetensors",
        "extractor_ckpt": {
            "dynamics": "./checkpoints/stable_audio_new_fast_all_v2/checkpoint-44000/model_1.safetensors",
            "melody": "./checkpoints/stable_audio_new_fast_all_v2/checkpoint-44000/model.safetensors",
            "rhythm": "./checkpoints/stable_audio_new_fast_all_v2/checkpoint-44000/model_2.safetensors",
        },
        ###############
        "GPU_id": "1",
        "attn_processor_type": "rotary", # Currently no other available.
        "apadapter": True, # True for MuseControlLite, False for original Stable-audio
        "ap_scale": 1.0, # recommend 1.0 for MuseControlLite, other values are not tested
        "guidance_scale_text": 7.0,
        "guidance_scale_con": 1.0, # Note that if guidance scale is too large, the audio quality will be bad. 
        "denoise_step": 50,
        "sigma_min": 0.3, # sigma_min and sigma_max are for the scheduler.
        "sigma_max": 500,  # Note that if sigma_max is too large or too small, the "audio condition generation" will be bad. 
        "weight_dtype": "fp16",
        "negative_text_prompt": "",

        # The below two mask should complementary, which means every time slice shouldn't receive both audio and music attribute condition.
        # Only one of use_audio_mask and use_musical_attribute_mask should be set to True.
        "use_audio_mask": False,
        "audio_mask_start_seconds": 24,
        "audio_mask_end_seconds": 2097152 / 44100, # Maximum duration for stable-audio is 2097152 / 44100 seconds
        "use_musical_attribute_mask": False,
        "musical_attribute_mask_start_seconds": 24,
        "musical_attribute_mask_end_seconds": 2097152 / 44100 ,

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
                # "",
                # "",
                "A heartfelt, warm acoustic guitar performance, evoking a sense of tenderness and deep emotion, with a melody that truly resonates and touches the heart.",     
                "A vibrant MIDI electronic composition with a hopeful and optimistic vibe.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "Hopeful instrumental with guitar being the lead and tabla used for percussion in the middle giving a feeling of going somewhere with positive outlook.",
                "A string ensemble opens the track with legato, melancholic melodies. The violins and violas play beautifully, while the cellos and bass provide harmonic support for the moving passages. The overall feel is deeply melancholic, with an emotionally stirring performance that remains harmonious and a sense of clearness.",
                "An exceptionally harmonious string performance with a lively tempo in the first half, transitioning to a gentle and beautiful melody in the second half. It creates a warm and comforting atmosphere, featuring cellos and bass providing a solid foundation, while violins and violas showcase the main theme, all without any noise, resulting in a cohesive and serene sound.",
                ]
    }
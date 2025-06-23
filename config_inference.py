def get_config():
    return {
        "condition_type": ["melody_stereo", "audio"], #  you can choose any combinations in the two sets: ["dynamics", "rhythm", "melody_mono", "audio"],  ["melody_stereo", "audio"]

        "output_dir": "./generated_audio/test",

        "GPU_id": "0",

        "apadapter": True, # True for MuseControlLite, False for original Stable-audio

        "ap_scale": 1.0, # recommend 1.0 for MuseControlLite, other values are not tested

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 2.0, # The separated guidance for Musical attribute condition
        
        "guidance_scale_audio": 0.5,
        
        "denoise_step": 50,

        "sigma_min": 0.3, # sigma_min and sigma_max are for the scheduler.

        "sigma_max": 500,  # Note that if sigma_max is too large or too small, the "audio condition generation" will be bad.

        "weight_dtype": "fp16", # fp16 and fp32 sounds quiet the same.

        "negative_text_prompt": "Low qualiy, noise",

        ###############
        "use_audio_mask": True, # Turn true to mask a portion of audio, enabling audio inpainting and outpainting. This will be automaticaly set to true if given both audio and musical attribute conditions

        "audio_mask_start_seconds": 24,

        "audio_mask_end_seconds": 2097152 / 44100, # Maximum duration for stable-audio is 2097152 / 44100 seconds

        "use_musical_attribute_mask": False, # Set to true, if you want to mask melody, rhythm, dynamics.

        "musical_attribute_mask_start_seconds": 24,

        "musical_attribute_mask_end_seconds": 2097152 / 44100 ,

        "buffer_seconds": 3, 
        ###############

        "no_text": False, # Optional, set to true if no text prompt is needed (possible for audio inpainting or outpainting)

        "show_result_and_plt": True,

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
                "A heartfelt, warm acoustic guitar performance, evoking a sense of tenderness and deep emotion, with a melody that truly resonates and touches the heart.",     
                "A vibrant MIDI electronic composition with a hopeful and optimistic vibe.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "Hopeful instrumental with guitar being the lead and tabla used for percussion in the middle giving a feeling of going somewhere with positive outlook.",
                "A string ensemble opens the track with legato, melancholic melodies. The violins and violas play beautifully, while the cellos and bass provide harmonic support for the moving passages. The overall feel is deeply melancholic, with an emotionally stirring performance that remains harmonious and a sense of clearness.",
                "An exceptionally harmonious string performance with a lively tempo in the first half, transitioning to a gentle and beautiful melody in the second half. It creates a warm and comforting atmosphere, featuring cellos and bass providing a solid foundation, while violins and violas showcase the main theme, all without any noise, resulting in a cohesive and serene sound.",
                ],

        ########## adapters avilable ############
        # MuseControlLite_inference_all.py will automaticaly choose the most suitable model according to the condition type:
        ###############
        # Works for condition ["dynamics", "rhythm", "melody_mono"]
        "transformer_ckpt_musical": "./checkpoints/woSDD-all/model_3.safetensors",
        
        "extractor_ckpt_musical": {
            "dynamics": "./checkpoints/woSDD-all/model_1.safetensors",
            "melody": "./checkpoints/woSDD-all/model.safetensors",
            "rhythm": "./checkpoints/woSDD-all/model_2.safetensors",
        },
        ###############

        # Works for ['audio], it works without a feature extractor, and could cooperate with other adapters
        #################
        "audio_transformer_ckpt": "./checkpoints/70000_Audio/model.safetensors",

        # Specialized for ['melody_stereo']
        ###############
        "transformer_ckpt_melody_stero": "./checkpoints/70000_Melody_stereo/model_1.safetensors",

        "extractor_ckpt_melody_stero": {
            "melody": "./checkpoints/70000_Melody_stereo/model.safetensors",
        },
        ###############

        # Specialized for ['melody_mono']
        ###############
        "transformer_ckpt_melody_mono": "./checkpoints/40000_Melody_mono/model_1.safetensors",

        "extractor_ckpt_melody_mono": {
            "melody": "./checkpoints/40000_Melody_mono/model.safetensors",
        },
        ###############
    }
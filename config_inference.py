def get_config():
    return {
        "condition_type": ["audio"], #  you can choose any combinations in the two sets: ["dynamics", "rhythm", "melody_mono", "audio"],  ["melody_stereo", "audio"]
                                    # When using audio, is recommend to use empty string "" as prompt
        "output_dir": "./generated_audio/outpainting",

        "GPU_id": "0",

        "apadapter": True, # True for MuseControlLite, False for original Stable-audio

        "ap_scale": 1.0, # recommend 1.0 for MuseControlLite, other values are not tested

        "guidance_scale_text": 7.0,

        "guidance_scale_con": 1.5, # The separated guidance for Musical attribute condition
        
        "guidance_scale_audio": 1.0,
        
        "denoise_step": 50,

        "sigma_min": 0.3, # sigma_min and sigma_max are for the scheduler.

        "sigma_max": 500,  # Note that if sigma_max is too large or too small, the "audio condition generation" will be bad.

        "weight_dtype": "fp16", # fp16 and fp32 sounds quiet the same.

        "negative_text_prompt": "",

        ###############

        "audio_mask_start_seconds": 14, # Apply mask to musical attributes choose only one mask to use, it automatically generates a complemetary mask to the other condition

        "audio_mask_end_seconds": 47, 

        "musical_attribute_mask_start_seconds": 0, # 'Apply mask to audio condition, choose only one mask to use, it automatically generates a complemetary mask to the other condition'

        "musical_attribute_mask_end_seconds": 0,

        ###############

        "no_text": False, # Optional, set to true if no text prompt is needed (possible for audio inpainting or outpainting)

        "show_result_and_plt": False,

        "audio_files": [
            "melody_condition_audio/49_piano.mp3",
            "melody_condition_audio/49_piano.mp3",
            "melody_condition_audio/49_piano.mp3",
            "melody_condition_audio/322_piano.mp3",
            "melody_condition_audio/322_piano.mp3",
            "melody_condition_audio/322_piano.mp3",
            "melody_condition_audio/610_bass.mp3",
            "melody_condition_audio/610_bass.mp3",
            "melody_condition_audio/785_piano.mp3",
            "melody_condition_audio/785_piano.mp3",
            "melody_condition_audio/933_string.mp3",
            "melody_condition_audio/933_string.mp3",
            "melody_condition_audio/6_uke_12.wav",
            "melody_condition_audio/6_uke_12.wav",
            "melody_condition_audio/57_jazz.mp3",
            "melody_condition_audio/703_mideast.mp3",

        ],
        # "audio_files": [
        #     "SDD_nosinging/SDD_audio/34/1004034.mp3",
        #     "original_15s/original_9.wav",
        #     "original_15s/original_10.wav",
        #     "original_15s/original_11.wav",
        #     "original_15s/original_15.wav",
        #     "original_15s/original_16.wav",
        #     "original_15s/original_21.wav",
        #     "original_15s/original_25.wav",
        # ],

        "text": [
                "Electronic music that has a constant melody throughout with accompanying instruments used to supplement the melody which can be heard in possibly a casual setting",
                "A heartfelt, warm acoustic guitar performance, evoking a sense of tenderness and deep emotion, with a melody that truly resonates and touches the heart.",     
                "A vibrant MIDI electronic composition with a hopeful and optimistic vibe.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "This track composed of electronic instruments gives a sense of opening and clearness.",
                "Hopeful instrumental with guitar being the lead and tabla used for percussion in the middle giving a feeling of going somewhere with positive outlook.",
                "A string ensemble opens the track with legato, melancholic melodies. The violins and violas play beautifully, while the cellos and bass provide harmonic support for the moving passages. The overall feel is deeply melancholic, with an emotionally stirring performance that remains harmonious and a sense of clearness.",
                "An exceptionally harmonious string performance with a lively tempo in the first half, transitioning to a gentle and beautiful melody in the second half. It creates a warm and comforting atmosphere, featuring cellos and bass providing a solid foundation, while violins and violas showcase the main theme, all without any noise, resulting in a cohesive and serene sound.",
                "Pop solo piano instrumental song. Simple harmony and emotional theme. Makes you feel nostalgic and wanting a cup of warm tea sitting on the couch while holding the person you love.",
                "A whimsical string arrangement with rich layers, featuring violins as the main melody, accompanied by violas and cellos. The light, playful melody blends harmoniously, creating a sense of clarity.",
                "An instrumental piece primarily featuring acoustic guitar, with a lively and nimble feel. The melody is bright, delivering an overall sense of joy.",
                "A joyful saxophone performance that is smooth and cohesive, accompanied by cello. The first half features a relaxed tempo, while the second half picks up with an upbeat rhythm, creating a lively and energetic atmosphere. The overall sound is harmonious and clear, evoking feelings of happiness and vitality.",
                "A cheerful piano performance with a smooth and flowing rhythm, evoking feelings of joy and vitality.",
                "An instrumental piece primarily featuring piano, with a lively rhythm and cheerful melodies that evoke a sense of joyful childhood playfulness. The melodies are clear and bright.",
                "fast and fun beat-based indie pop to set a protagonist-gets-good-at-x movie montage to.",
                "A lively 70s style British pop song featuring drums, electric guitars, and synth violin. The instruments blend harmoniously, creating a dynamic, clean sound without any noise or clutter.",
                "A soothing acoustic guitar song that evokes nostalgia, featuring intricate fingerpicking. The melody is both sacred and mysterious, with a rich texture."
                ],

        ########## adapters avilable ############
        # We trained 4 set of adapters:
        # 1. with conditions ["melody_mono", "dynamics", "rhythm"]
        # 2. with conditions ["melody_mono"]
        # 3. with conditions ["melody_stereo"]
        # 3. with conditions ["audio"]
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
        "audio_transformer_ckpt": "./model.safetensors",

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
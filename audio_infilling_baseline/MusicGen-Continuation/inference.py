## inference for music continuation
## reference: https://github.com/facebookresearch/audiocraft/blob/main/demos/musicgen_demo.ipynb

import argparse
import torch
import torchaudio
from audiocraft.models import MusicGen

parser = argparse.ArgumentParser(description="MusicGen Inference")
parser.add_argument('--model_size', dest='model_size', type=str, default='small', 
                    choices=['small', 'medium', 'large'], help='MusicGen model size')
# generation parameters
parser.add_argument('--use_sampling', dest='use_sampling', action='store_true', 
                    help='enable argmax decoding')
parser.add_argument('--top_k', dest='top_k', type=int, default=250,
                    help='top_k used for sampling')
parser.add_argument('--top_p', dest='top_p', type=float, default=0.0,
                    help='top_p used for sampling')
parser.add_argument('--temperature', dest='temperature', type=float, default=1.0,
                    help='softmax temperature')
parser.add_argument('--duration', dest='duration', type=float, default=30.0,
                    help='duration of the generated waveform')
parser.add_argument('--cfg_coef', dest='cfg_coef', type=float, default=3.0,
                    help='coefficient used for classifier free guidance')
# arguments for music continuation
parser.add_argument('--reference_audio', dest='reference_audio', required=True, 
                    type=str, help='reference audio')
parser.add_argument('--reference_duration', dest='reference_duration', type=float, 
                    default=15.0, help='duration of the reference audio')
parser.add_argument('--text_prompt', dest='text_prompt', type=str,
                    default=None, help='text prompt')
parser.add_argument('--progress', dest='progress', action='store_true', 
                    help='display progress of the generation process')
parser.add_argument('--return_tokens', dest='return_tokens', action='store_true', 
                    help='return generated tokens')
parser.add_argument('--output', dest='output', type=str, default='musicgen-cont.mp3',
                    help='output path')

def main():
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_ckpt = f"facebook/musicgen-{args.model_size}"
    model = MusicGen.get_pretrained(model_ckpt)

    ## configure generation parameters
    model.set_generation_params(
        use_sampling=args.use_sampling,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        duration=args.duration,
        cfg_coef=args.cfg_coef
    )

    ## music continuation generation
    waveform, sr = torchaudio.load(args.reference_audio)
    # trim the reference audio to the desired length
    waveform = waveform[..., :int(args.reference_duration * sr)]  

    output = model.generate_continuation(
        prompt=waveform, 
        prompt_sample_rate=sr,
        descriptions=[args.text_prompt], 
        progress=args.progress, 
        return_tokens=args.return_tokens
    )
    if (args.return_tokens):
        result = output[0].detach().cpu().squeeze(0)
    else:
        result = output.detach().cpu().squeeze(0)
    
    torchaudio.save(uri=args.output, src=result, sample_rate=32000)

if __name__ == '__main__':
    main()
import os
import glob
import numpy as np
import libfmp.c4

# — your helper functions —
def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
    taper = np.sqrt(1/2) / (L * var)
    axis = np.arange(-L, L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel

def compute_novelty_ssm(S, kernel=None, L=10, var=0.5, exclude=False):
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    S_padded = np.pad(S, L, mode='constant')
    for n in range(N):
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        nov[:L] = 0
        nov[-L:] = 0
    return nov

# — batch parameters —
input_folder = "/home/fundwotsai/Music-Controlnet-light/audio_outpainting/"
L_kernel   = 3
boundary   = 10

# prepare output file
out_txt = os.path.join(input_folder, "novelty_means.txt")
with open(out_txt, "w", encoding="utf-8") as fout:
    fout.write("subfolder,mean_novelty_value\n")

    # for each immediate subfolder
    for entry in sorted(os.listdir(input_folder)):
        subdir = os.path.join(input_folder, entry)
        if not os.path.isdir(subdir):
            continue

        # collect audio files in this subfolder (non-recursive)
        audio_paths = (
            glob.glob(os.path.join(subdir, "*.mp3")) +
            glob.glob(os.path.join(subdir, "*.wav"))
        )
        if not audio_paths:
            # skip empty folders
            continue

        nov_vals = []
        for fn_wav in audio_paths:
            # compute self-similarity matrix
            x, x_dur, X, Fs_X, S, I = libfmp.c4.compute_sm_from_filename(
                fn_wav, L=81, H=10, L_smooth=1, thresh=1
            )
            # novelty curve
            nov = compute_novelty_ssm(S, L=L_kernel, exclude=True)
            # your feature at `boundary`
            val = nov[boundary] - 0.5*nov[boundary-1] - 0.5*nov[boundary+1]
            nov_vals.append(val)

        # mean over all files in this subfolder
        mean_val = np.mean(nov_vals)
        fout.write(f"{entry},{mean_val:.6f}\n")

print(f"✔️ Done! Per-folder means in: {out_txt}")

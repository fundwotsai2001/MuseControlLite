import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import librosa

def compute_music_represent(audio, sr):
    # audio shape: (samples,) for mono or (channels, samples) for stereo
    # For your function, expects 1D tensor, so process per channel
    filter_y = torchaudio.functional.highpass_biquad(audio, sr, 261.6)
    fmin = librosa.midi_to_hz(0)
    cqt_spec = librosa.cqt(y=filter_y.numpy(), fmin=fmin, sr=sr, n_bins=128, bins_per_octave=12, hop_length=512)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)
    return cqt_db
def keep_top4_pitches_per_channel(cqt_db):
    """
    cqt_db is assumed to have shape: (2, 128, time_frames).
    We return a combined 2D array of shape (128, time_frames)
    where only the top 4 pitch bins in each channel are kept
    (for a total of up to 8 bins per time frame).
    """
    # Parse shapes
    num_channels, num_bins, num_frames = cqt_db.shape
    
    # Initialize an output array that combines both channels
    # and has zeros everywhere initially
    combined = np.zeros((num_bins, num_frames), dtype=cqt_db.dtype)
    
    for ch in range(num_channels):
        for t in range(num_frames):
            # Find the top 4 pitch bins for this channel at frame t
            # argsort sorts ascending; we take the last 4 indices for top 4
            top4_indices = np.argsort(cqt_db[ch, :, t])[-4:]
            
            # Copy their values into the combined array
            # We add to it in case there's overlap between channels
            combined[top4_indices, t] = 1
    return combined
def plot_stereo_cqt(audio_stereo, sr):
    """
    audio_stereo: torch.Tensor with shape (2, samples)
    sr: sample rate
    """
    assert audio_stereo.shape[0] == 2, "Input audio should be stereo with shape (2, samples)"

    cqt = compute_music_represent(audio_stereo, sr)
    # cqt_right = compute_music_represent(audio_stereo[1], sr)
    melody_condition = keep_top4_pitches_per_channel(cqt)
    print(melody_condition.shape)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(melody_condition, aspect='auto', origin='lower', cmap='plasma')
    ax.set_title("Top 4 Pitches Per Channel (Binary Mask)")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("CQT Pitch Bin (Low to High)")
    plt.colorbar(im, ax=ax, label="Presence")
    plt.tight_layout()
    plt.savefig("cqt.png")

# Example usage:
audio, sr = torchaudio.load("/home/fundwotsai/MuseControlLite/melody_condition_audio/49_piano.mp3")
plot_stereo_cqt(audio, sr)

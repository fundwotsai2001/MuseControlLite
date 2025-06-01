import os
import torchaudio
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
from multiprocessing import Pool, cpu_count
import librosa
import typing as tp
from einops import rearrange
from librosa import filters
import torch
from torch import nn
import torchaudio
import typing as tp
import torch.nn.functional as F
import scipy.signal as signal
from torchaudio import transforms as T
import torch
import torchaudio
import librosa
import numpy as np

def compute_melody_v2(audio, sr):
    sample_rate = 44100

    # Load audio file
    wav, sr = torchaudio.load(audio)
    if sr != sample_rate:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        wav = resample(wav)
    # Truncate or pad the audio to 2097152 samples
    target_length = 2097152
    if wav.size(1) > target_length:
        # Truncate the audio if it is longer than the target length
        wav = wav[:, :target_length]
    elif wav.size(1) < target_length:
        # Pad the audio with zeros if it is shorter than the target length
        padding = target_length - wav.size(1)
        wav = torch.cat([wav, torch.zeros(wav.size(0), padding)], dim=1)
    filter_y = torchaudio.functional.highpass_biquad(wav, sr, 261.6)

    fmin = librosa.midi_to_hz(0)
    cqt_list = []
    for ch in range(filter_y.shape[0]):
        cqt_spec = librosa.cqt(y=filter_y[ch].numpy(), fmin=fmin, sr=sr, n_bins=128, bins_per_octave=12, hop_length=512)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt_spec), ref=np.max)  # (freq_bins, time_frames)
        cqt_list.append(cqt_db)

    cqt_ch0 = cqt_list[0]
    cqt_ch1 = cqt_list[1]

    time_frames = cqt_ch0.shape[1]

    # 對每個時間點排序，找前4大bin的索引 (頻率維度)
    # 因為是dB值，數值越大代表能量越高
    top4_idx_ch0 = np.argsort(cqt_ch0, axis=0)[-4:, :]  # shape (4, time_frames)
    top4_idx_ch1 = np.argsort(cqt_ch1, axis=0)[-4:, :]  # shape (4, time_frames)

    # 準備結果陣列 shape (8, time_frames)
    interleaved = np.zeros((8, time_frames), dtype=cqt_ch0.dtype)

    # 對每個時間幀取值，交錯放置
    for t in range(time_frames):
        # 取出兩個channel當前時間點 top4 bin索引
        idx0 = top4_idx_ch0[:, t]  # (4,)
        idx1 = top4_idx_ch1[:, t]  # (4,)

        # 取能量值
        vals0 = cqt_ch0[idx0, t]
        vals1 = cqt_ch1[idx1, t]

        # 交錯放入結果陣列
        interleaved[0::2, t] = vals0
        interleaved[1::2, t] = vals1

    return torch.tensor(interleaved)

def compute_music_represent(audio, sr):
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
def compute_melody(input_audio):
    # Initialize parameters
    sample_rate = 44100

    # Load audio file
    wav, sr = torchaudio.load(input_audio)
    if sr != sample_rate:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        wav = resample(wav)
    # Truncate or pad the audio to 2097152 samples
    target_length = 2097152
    if wav.size(1) > target_length:
        # Truncate the audio if it is longer than the target length
        wav = wav[:, :target_length]
    elif wav.size(1) < target_length:
        # Pad the audio with zeros if it is shorter than the target length
        padding = target_length - wav.size(1)
        wav = torch.cat([wav, torch.zeros(wav.size(0), padding)], dim=1)
    melody = compute_music_represent(wav, 44100)
    melody = keep_top4_pitches_per_channel(melody)    
    return melody

def compute_dynamics(audio_file, hop_length=160, target_sample_rate=44100, cut=False):
    """
    Compute the dynamics curve for a given audio file.
    
    Args:
        audio_file (str): Path to the audio file.
        window_length (int): Length of FFT window for computing the spectrogram.
        hop_length (int): Number of samples between successive frames.
        smoothing_window (int): Length of the Savitzky-Golay filter window.
        polyorder (int): Polynomial order of the Savitzky-Golay filter.

    Returns:
        dynamics_curve (numpy.ndarray): The computed dynamic values in dB.
    """
    # Load audio file
    waveform, original_sample_rate = torchaudio.load(audio_file)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    if cut:
        waveform = waveform[:, :2097152]
    # Ensure waveform has a single channel (e.g., select the first channel if multi-channel)
    waveform = waveform.mean(dim=0, keepdim=True)  # Mix all channels into one
    waveform = waveform.clamp(-1, 1).numpy()
    
    S = np.abs(librosa.stft(waveform, n_fft=1024, hop_length=hop_length))
    mel_filter_bank = librosa.filters.mel(sr=target_sample_rate, n_fft=1024, n_mels=64, fmin=0, fmax=8000)
    S = np.dot(mel_filter_bank, S)
    energy = np.sum(S**2, axis=0)
    dynamics_db = np.clip(energy, 1e-6, None)
    dynamics_db = librosa.amplitude_to_db(energy, ref=np.max).squeeze(0)
    smoothed_dynamics = savgol_filter(dynamics_db, window_length=279, polyorder=1)
    # print(smoothed_dynamics.shape)
    return smoothed_dynamics
def extract_melody_one_hot(audio_path,
                           sr=44100,
                           cutoff=261.2, 
                           win_length=2048,
                           hop_length=256):
    """
    Extract a one-hot chromagram-based melody from an audio file (mono).
    
    Parameters:
    -----------
    audio_path : str
        Path to the input audio file.
    sr : int
        Target sample rate to resample the audio (default: 44100).
    cutoff : float
        The high-pass filter cutoff frequency in Hz (default: Middle C ~ 261.2 Hz).
    win_length : int
        STFT window length for the chromagram (default: 2048).
    hop_length : int
        STFT hop length for the chromagram (default: 256).
    
    Returns:
    --------
    one_hot_chroma : np.ndarray, shape=(12, n_frames)
        One-hot chromagram of the most prominent pitch class per frame.
    """
    # ---------------------------------------------------------
    # 1. Load audio (Torchaudio => shape: (channels, samples))
    # ---------------------------------------------------------
    audio, in_sr = torchaudio.load(audio_path)

    # Convert to mono by averaging channels: shape => (samples,)
    audio_mono = audio.mean(dim=0)

    # Resample if necessary
    if in_sr != sr:
        resample_tf = T.Resample(orig_freq=in_sr, new_freq=sr)
        audio_mono = resample_tf(audio_mono)

    # Convert torch.Tensor => NumPy array: shape (samples,)
    y = audio_mono.numpy()

    # ---------------------------------------------------------
    # 2. Design & apply a high-pass filter (Butterworth, order=2)
    # ---------------------------------------------------------
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(N=2, Wn=norm_cutoff, btype='high', analog=False)
    
    # filtfilt expects shape (n_samples,) for 1D
    y_hp = signal.filtfilt(b, a, y)

    # ---------------------------------------------------------
    # 3. Compute the chromagram (librosa => shape: (12, n_frames))
    # ---------------------------------------------------------
    chroma = librosa.feature.chroma_stft(
        y=y_hp,
        sr=sr,
        n_fft=win_length,      # Usually >= win_length
        win_length=win_length,
        hop_length=hop_length
    )

    # ---------------------------------------------------------
    # 4. Convert chromagram to one-hot via argmax along pitch classes
    # ---------------------------------------------------------
    # pitch_class_idx => shape=(n_frames,)
    pitch_class_idx = np.argmax(chroma, axis=0)

    # Make a zero array of the same shape => (12, n_frames)
    one_hot_chroma = np.zeros_like(chroma)

    # For each frame (column in chroma), set the argmax row to 1
    one_hot_chroma[pitch_class_idx, np.arange(chroma.shape[1])] = 1.0
    
    return one_hot_chroma
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

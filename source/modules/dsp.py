import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import uniform_filter1d
import os

def detect_event_bounds(audio_array,
                       sr=22050,
                       smooth_ms=20,
                       threshold_ratio=0.2,
                       min_gap_s=0.05,
                       min_call_s=0.03):
    """
    Detects the main bird call (onset, offset) in a clean recording.
    Small gaps between energy bursts are merged if shorter than `min_gap_s`.

    Returns:
        (onset_time_s, offset_time_s)
        or None if no prominent call is found.
    """
    # 1. Load audio
    y = audio_array

    # 2. Short-time energy (RMS)
    hop_length = int(0.005 * sr)  # 5 ms hop
    frame_length = int(0.02 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # 3. Smooth the energy envelope
    win = int(smooth_ms / 5) or 1
    smooth_rms = np.convolve(rms, np.ones(win)/win, mode='same')

    # 4. Threshold at a fraction of the maximum energy
    thresh = threshold_ratio * np.max(smooth_rms)
    mask = smooth_rms > thresh

    # 5. Convert mask to contiguous regions
    diff = np.diff(mask.astype(int))
    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1
    if mask[0]:
        onsets = np.r_[0, onsets]
    if mask[-1]:
        offsets = np.r_[offsets, len(mask)]

    # Ensure indices are within range
    onsets = np.clip(onsets, 0, len(times) - 1)
    offsets = np.clip(offsets, 0, len(times) - 1)

    if len(onsets) == 0:
        return None

    # 6. Merge regions separated by short gaps
    merged_onsets = [onsets[0]]
    merged_offsets = []
    for i in range(1, len(onsets)):
        gap = times[onsets[i]] - times[offsets[i-1]]
        if gap <= min_gap_s:
            # Merge with previous
            continue
        else:
            merged_offsets.append(offsets[i-1])
            merged_onsets.append(onsets[i])
    merged_offsets.append(offsets[-1])

    # 7. Keep only regions longer than min_call_s
    events = [(times[o], times[f]) for o, f in zip(merged_onsets, merged_offsets)
              if (times[f] - times[o]) >= min_call_s]
    if not events:
        return None

    # 8. Return all regions in s
    return events

def get_longest_event(events):
    # Return the longest region (main call)
    durations = [f - o for o, f in events]
    i = np.argmax(durations)
    onset, offset = events[i]
    return onset, offset

def merge_gaps(events, min_gap_s, sampling_rate):
    sr = sampling_rate
    onsets, offsets = zip(*events)
    print(onsets)
    merged_onsets = [onsets[0]]
    merged_offsets = []
    for i in range(1, len(onsets)):
        gap = onsets[i] - offsets[i-1]
        if gap <= min_gap_s:
            # Merge with previous
            continue
        else:
            merged_offsets.append(offsets[i-1])
            merged_onsets.append(onsets[i])
    merged_offsets.append(offsets[-1])

    events = [(o, f) for o, f in zip(merged_onsets, merged_offsets)]
    return events

def num_samples_to_duration_s(num_samples, sampling_rate):
    return num_samples / sampling_rate

def duration_s_to_num_samples(duration_s, sampling_rate):
    return duration_s * sampling_rate

def silence_gaps(audio_array, sampling_rate, events):
    # get sample precise on- and offsets
    event_samples = [(duration_s_to_num_samples(onset, sampling_rate), duration_s_to_num_samples(offset, sampling_rate)) for (onset, offset) in events]
    onsets, offsets = zip(*event_samples)

    # silence gaps
    clean_audio = np.zeros(len(audio_array))
    copy_values = False

    for i in range(len(audio_array)):

        if i in onsets:
            copy_values = True
        if i in offsets:
            copy_values = False

        if copy_values:
            clean_audio[i] = audio_array[i]

    return clean_audio

def stft_mask_bandpass(y, sr,
                       n_fft=1024, hop_length=None,
                       collapse='max', smooth_bins=5,
                       low_pct=5, high_pct=95,
                       edge_bins=5,   # soft edge width in freq bins
                       events = None # list of (onset_s, offset_s) or None -> whole file
                       #segments=None  # list of (start_sample, end_sample) or None -> whole file
                      ):
    """
    Remove energy outside percentile-based frequency bands (per segment or globally).
    Returns:
        y_out: filtered waveform (same length as input)
        bounds_list: list of tuples (segment_start_sample, segment_end_sample, f_low, f_high)
    """
    if hop_length is None:
        hop_length = n_fft // 4

    # STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    mags = np.abs(S)
    phase = np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    n_bins, n_frames = mags.shape

    # If no segments provided, treat the whole file as one segment
    if events is None:
        segments = None
    else:
        segments = [(duration_s_to_num_samples(onset, sr), duration_s_to_num_samples(offset, sr)) for (onset, offset) in events]

    if segments is None:
        segments = [(0, len(y))]

    # Create mask initialized zeros and init bounds list
    mask = np.zeros_like(mags)
    bounds_list = []

    for (start_sample, end_sample) in segments:
        # map sample indices to frame indices
        t0 = int(np.floor(start_sample / float(hop_length)))
        t1 = int(np.ceil(end_sample / float(hop_length)))
        t0 = max(0, t0)
        t1 = min(n_frames, t1)

        if t0 >= t1:
            continue

        # collapse magnitude across the selected frames to get a mean spectrum
        block = mags[:, t0:t1]
        if collapse == 'max':
            spec = np.max(block, axis=1)
        elif collapse == 'median':
            spec = np.median(block, axis=1)
        else:
            spec = np.mean(block, axis=1)

        # smooth and normalize
        if smooth_bins and smooth_bins > 1:
            spec = uniform_filter1d(spec, size=smooth_bins)
        spec = np.maximum(spec, 0.0)
        total = spec.sum()
        if total <= 0:
            continue
        spec_norm = spec / total
        cumsum = np.cumsum(spec_norm)

        # find percentile bounds
        f_low = np.interp(low_pct/100.0, cumsum, freqs)
        f_high = np.interp(high_pct/100.0, cumsum, freqs)

        # Store the bounds
        bounds_list.append((num_samples_to_duration_s(start_sample, sr), num_samples_to_duration_s(end_sample, sr), float(f_low), float(f_high)))

        # convert to bin indices
        idx_low = np.searchsorted(freqs, f_low)
        idx_high = np.searchsorted(freqs, f_high)

        # build mask for frames t0:t1
        for f_idx in range(n_bins):
            if idx_low <= f_idx <= idx_high:
                mask[f_idx, t0:t1] = 1.0

        # apply soft edges (linear ramp) around idx_low/idx_high
        if edge_bins > 0:
            for k in range(1, edge_bins+1):
                if idx_low - k >= 0:
                    ramp = (edge_bins - (k-1)) / float(edge_bins+1)  # simple ramp
                    mask[idx_low - k, t0:t1] = np.maximum(mask[idx_low - k, t0:t1], ramp)
                if idx_high + k < n_bins:
                    ramp = (edge_bins - (k-1)) / float(edge_bins+1)
                    mask[idx_high + k, t0:t1] = np.maximum(mask[idx_high + k, t0:t1], ramp)

    # Apply mask (elementwise) to magnitude and rebuild complex STFT
    S_masked = mask * mags * np.exp(1j * phase)

    # ISTFT
    y_out = librosa.istft(S_masked, hop_length=hop_length, win_length=n_fft, length=len(y))

    return y_out, bounds_list

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import ast

def plot_save_mel_spectrogram(audio_array, sampling_rate, filename=None, details=None, output_dir=None):
    """
    Takes audio_array and sampling_rate as input. Computes Mel spectrogram, plot waveform + spectrogram,
    and saves figure to output_dir if provided.

    Parameters
    ----------
    audio : huggingface Audio object
    """

    # Enforce mono 
    if not audio_array.ndim == 1:
        audio_array = np.mean(audio_array, axis=0)

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sampling_rate, fmax=8000)
    S_dB = librosa.power_to_db(mel_spec, ref=np.max)

    # --- Create figure using GridSpec for precise layout ---
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 3], figure=fig)

    # Axes
    ax_wave = fig.add_subplot(gs[0, 0])  # waveform
    ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_wave)  # spectrogram
    cax = fig.add_subplot(gs[1, 1])  # colorbar

    # --- Plot waveform ---
    times = np.arange(audio_array.size) / sampling_rate
    ax_wave.plot(times, audio_array, color="gray")
    ax_wave.set_ylabel("Amplitude")

    title = ""
    if filename:
        title += filename
    if details:
        title += f" [{details}]"
    ax_wave.set_title(title)

    ax_wave.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax_wave.get_xticklabels(), visible=False)  # hide x labels on top plot

    # --- Plot spectrogram ---
    img = librosa.display.specshow(
        S_dB,
        x_axis="time",
        y_axis="mel",
        sr=sampling_rate,
        fmax=8000,
        ax=ax_spec,
    )
    fig.colorbar(img, cax=cax, format="%+2.0f dB", label="dB")
    ax_spec.set_xlabel("Time [s]")
    ax_spec.set_ylabel("Mel frequency [Hz]")

    plt.tight_layout()
    plt.show()

    # --- Save figure ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if details:
            save_path = output_dir + f"{Path(filename).stem}_{details}.png"
        else:
            save_path = output_dir + f"{Path(filename).stem}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"Saved Mel spectrogram to: {save_path}")

    
def segment_audio(audio_array, sampling_rate, segment_length, keep_incomplete=False):
    """
    Splits an audio array into fixed-length segments.
    If keep_incomplete=True, the last (possibly shorter) segment is kept.
    Each segment is returned along with its start and end timestamps (in seconds).

    Parameters:
        audio_array (np.ndarray): The input audio array (1D or 2D).
        sampling_rate (int): Sampling rate of the audio in Hz.
        segment_length (float): Desired length of each segment in seconds.
        keep_incomplete (bool): Whether to keep the final shorter segment.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - "segment" (np.ndarray): The audio segment.
            - "start_time" (float): Start time in seconds.
            - "end_time" (float): End time in seconds.
    """
    samples_per_segment = int(sampling_rate * segment_length)
    total_samples = len(audio_array)
    segments = []

    for start in range(0, total_samples, samples_per_segment):
        end = start + samples_per_segment
        segment = audio_array[start:end]

        # Skip incomplete if not desired
        if not keep_incomplete and len(segment) < samples_per_segment:
            continue

        start_time = start / sampling_rate
        end_time = min(end, total_samples) / sampling_rate

        segments.append({
            "audio_array": segment,
            "start_time": start_time,
            "end_time": end_time
        })

    return segments

def remove_segments_without_events(segments, events):
    """
    Checks which audio segments contain at least one event.

    Parameters:
        segments (list[dict]): Each dict must have 'start_time' and 'end_time' (in seconds).
        events (list[tuple]): Each tuple is (event_start, event_end) in seconds.

    Returns:
        list[dict]: A filtered list of segments that contain at least one event.
    """
    if not events:
        return False

    def overlaps(seg_start, seg_end, evt_start, evt_end):
        # Two intervals overlap if they intersect at all
        return not (evt_end <= seg_start or evt_start >= seg_end)

    segments_with_event = []
    for seg in segments:
        seg_start = seg["start_time"]
        seg_end = seg["end_time"]

        # Check if any event overlaps this segment
        if not events: print('events = None')
        if any(overlaps(seg_start, seg_end, e_start, e_end) for e_start, e_end in events):
            segments_with_event.append(seg)

    return segments_with_event

import numpy as np

def pad_audio_end(audio: np.ndarray, sr: int, desired_length_s: float):
    """
    Pads an audio signal at the end to reach the desired length.

    Parameters
    ----------
    audio : np.ndarray
        Input audio array (1D or 2D with shape (channels, samples)).
    sr : int
        Sampling rate in Hz.
    desired_length_s : float
        Desired length of the audio in seconds.

    Returns
    -------
    padded_audio : np.ndarray
        Audio array padded at the end to the desired length.
    pad_end_s : float
        Amount of padding added at the end in seconds.
    """
    target_length = int(desired_length_s * sr)
    current_length = audio.shape[-1]

    if current_length >= target_length:
        # Trim if already longer than target
        return audio[..., :target_length], 0.0

    pad_end = target_length - current_length

    if audio.ndim == 1:
        padded_audio = np.pad(audio, (0, pad_end), mode='constant')
    else:
        padded_audio = np.pad(audio, ((0, 0), (0, pad_end)), mode='constant')

    pad_end_s = pad_end / sr
    return padded_audio, pad_end_s

def calculate_rms(audio_array, sampling_rate, event_bounds=None):
    """
    Compute the RMS of a signal within specified time bounds.

    Parameters
    ----------
    audio_array : np.ndarray
        1D numpy array containing audio samples (float).
    sampling_rate : int or float
        Sampling rate of the audio signal in Hz.
    event_bounds : list of tuple(float, float, float, float)
        List of events as (start_time, end_time, f_low, f_high). f_low and f_high are ignored.
        If None, RMS is computed over the entire signal.

    Returns
    -------
    rms_value : float
        RMS value of the signal within the combined event time windows.
    """

    if audio_array.ndim != 1:
        raise ValueError("audio_array must be a 1D numpy array.")

    if not event_bounds:
        event_bounds = [(0, len(audio_array), None, None)]

    total_energy = 0.0
    total_samples = 0

    for start_s, end_s, _, _ in event_bounds:
        # Convert times to sample indices
        start_i = int(np.floor(start_s * sampling_rate))
        end_i = int(np.ceil(end_s * sampling_rate))

        # Clamp indices to valid range
        start_i = max(0, min(len(audio_array), start_i))
        end_i = max(0, min(len(audio_array), end_i))

        # Extract the segment
        segment = audio_array[start_i:end_i]

        # Accumulate energy and sample count
        total_energy += np.sum(segment ** 2)
        total_samples += len(segment)

    if total_samples == 0:
        raise ValueError("No valid samples found within the provided event bounds.")

    rms_value = np.sqrt(total_energy / total_samples)
    return rms_value

import numpy as np

def normalize_to_dBFS(audio_array, target_dBFS, current_rms):
    """
    Normalize an audio signal to a target RMS level in dBFS.

    Parameters
    ----------
    audio_array : np.ndarray
        1D numpy array of audio samples (float, typically -1.0 to 1.0).
    target_dBFS : float
        Desired RMS level in dBFS (e.g., 0, -3, -6, etc.).
    current_rms : float
        The current RMS value of the signal (computed over the relevant region).

    Returns
    -------
    normalized_audio : np.ndarray
        The audio signal scaled so that its RMS equals the target dBFS value.
    """

    if audio_array.ndim != 1:
        raise ValueError("audio_array must be a 1D numpy array.")

    if current_rms <= 0:
        raise ValueError("current_rms must be positive and nonzero.")

    # Convert dBFS target to linear RMS
    target_rms_linear = dBFS_to_gain(target_dBFS)

    # Compute gain factor
    gain = target_rms_linear / current_rms

    # Apply gain
    normalized_audio = audio_array * gain

    return normalized_audio, gain

def dBFS_to_gain(dBFS_value):
    """
    Convert a dBFS (decibels relative to full scale) value to a linear amplitude gain factor.

    Parameters
    ----------
    dBFS_value : float or np.ndarray
        Level in decibels relative to full scale (dBFS). 
    Returns
    -------
    gain : float or np.ndarray
    """
    return 10 ** (dBFS_value / 20.0)


import numpy as np

def gain_to_dBFS(gain):
    """
    Convert a linear amplitude gain factor to decibels relative to full scale (dBFS).

    Parameters
    ----------
    gain : float or np.ndarray
        Linear amplitude gain factor.
        Must be positive and nonzero.

    Returns
    -------
    dBFS_value : float or np.ndarray
    """
    if np.any(gain <= 0):
        raise ValueError("Gain must be positive and nonzero to compute dBFS.")
    return 20.0 * np.log10(gain)

def extract_relevant_bounds(segment_start_time, segment_end_time, time_freq_bounds):
    """
    Extract and adjust time-frequency bounds relevant to a given audio segment.

    Parameters:
        segment_start_time (float): Start time of the segment (in seconds).
        segment_end_time (float): End time of the segment (in seconds).
        time_freq_bounds (list[tuple]): List of tuples
                                        (start_time, end_time, low_freq, high_freq)
                                        all in reference to the original file.

    Returns:
        list[tuple]: Relevant time-frequency bounds adjusted to be relative
                     to the start of the segment.
                     Format: (adj_start_time, adj_end_time, low_freq, high_freq)
    """
    relevant_bounds = []

    for start, end, low_f, high_f in time_freq_bounds:
        # Check overlap with segment
        if end <= segment_start_time or start >= segment_end_time:
            continue  # no overlap

        # Clip to the segment window
        clipped_start = max(start, segment_start_time)
        clipped_end = min(end, segment_end_time)

        # Shift so times are relative to the segment
        relative_start = clipped_start - segment_start_time
        relative_end = clipped_end - segment_start_time

        relevant_bounds.append((relative_start, relative_end, low_f, high_f))

    return relevant_bounds





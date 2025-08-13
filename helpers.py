import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from numpy.fft import fft, fftshift, fftfreq
from matplotlib import pyplot as plt
from pathlib import Path
import ffmpeg, librosa

def get_audio_data(mp4_file):
    probe = ffmpeg.probe(mp4_file)
    audio_stream = next((stream for stream in probe['streams'] 
                        if stream['codec_type'] == 'audio'), None)
    
    if audio_stream is None:
        raise ValueError(f"No audio stream found in {mp4_file}")
    
    sr = int(audio_stream['sample_rate'])
    channels = int(audio_stream['channels'])
    
    out, _ = (
        ffmpeg
        .input(mp4_file)
        .output('-', format='s16le', acodec='pcm_s16le')
        .run(capture_stdout=True, capture_stderr=True)
    )
    
    audio_array = np.frombuffer(out, dtype=np.int16)
    
    if channels > 1:
        audio_array = audio_array.reshape(-1, channels)
        ys = audio_array.mean(axis=1)
    else:
        ys = audio_array
    
    ys = ys / np.iinfo(np.int16).max
    
    num_samples = len(ys)
    ts = np.arange(num_samples) / sr
    
    return ys, ts, sr

def plot_fft(xs, ys):
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT")
    plt.xlim(0, 3000)
    plt.plot(xs, np.abs(ys))
    plt.savefig("FFT Plot")

def plotMagSpec(time_frames, freq_bins, frame_freq_amps):
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.pcolormesh(time_frames, freq_bins, frame_freq_amps, shading='gouraud')
    plt.title('Magnitude Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar()
    plt.ylim(0, 3000)

def get_fft(ys, sr):
    n = len(ys)
    
    window = np.hamming(n)

    ys *= window

    scale = 1.0 / np.sum(window) * n

    amps = fft(ys) / n
    amps[1:n//2] *= 2
    amps = amps[:n//2]
    amps *= scale
    freq_bins = fftfreq(n, d=1/sr)[:n//2]
    return freq_bins, amps

def differentiate(ys, xs):
    return np.diff(ys) / np.diff(xs)

def magnitude_spectrogram(ys, ts, sr):
    fr_len = .02 # 20ms frames
    hop_length = int(sr * fr_len)
    
    frame_freq_amps = []
    time_frames = []

    # Process first frame to get dimensions
    frame_ys = ys[0:hop_length]
    freq_bins, _ = get_fft(frame_ys, sr)
    
    for i in range(0, len(ys) - hop_length + 1, hop_length // 2):
        frame_ys = ys[i:i + hop_length].copy()
        _, amps = get_fft(frame_ys, sr)
        frame_freq_amps.append(np.abs(amps))
        time_frames.append(ts[i + hop_length // 2])
    
    time_frames = np.array(time_frames)
    frame_freq_amps = np.column_stack(frame_freq_amps)
    # plotMagSpec(time_frames=time_frames, freq_bins=freq_bins, frame_freq_amps=frame_freq_amps)
    return time_frames, frame_freq_amps, freq_bins

def compute_spectral_centroid(time_frames, freq_bins, frame_freq_amps):
    centroids = [0.0 for _ in range(len(time_frames))]

    assert len(freq_bins) == len(frame_freq_amps)
    assert len(time_frames) == len(frame_freq_amps[0]) - 1

    for i in range(len(time_frames)):
        amps_sum = 0; weighted_freq_sum = 0

        for j in range(len(freq_bins)):
            weight = abs(frame_freq_amps[j][i])
            weighted_freq_sum += freq_bins[j] * weight
            amps_sum += weight
        centroids[i] = weighted_freq_sum / amps_sum if amps_sum != 0 else 0
    
    centroids = gaussian_filter1d(centroids, sigma=1, truncate=3)

    return centroids
    
def compute_spectral_flux(time_frames, frame_freq_amps, sample_rate):
    # note that frame_freq_amps is already magnitude
    spec_flux = np.zeros(len(time_frames) - 1)

    assert(len(time_frames) == len(frame_freq_amps[0]))

    for t in range(1, len(time_frames)):
        flux = 0
        for f in range(len(frame_freq_amps)):
            # we half-wave rectify, so this only computes positive changes in
            flux += frame_freq_amps[f][t] - frame_freq_amps[f][t - 1]
        spec_flux[t - 1] = flux

    spec_flux = np.array(spec_flux)

    spec_flux = gaussian_filter1d(spec_flux, sigma=1, truncate=3)
    
    # apply filter to smooth. use median filter with window ~40 ms.
    # frame_hz = (time_frames[1] - time_frames[0]) * sample_rate
    # k = int(round(.04 * frame_hz)) | 1
    # spec_flux = medfilt(spec_flux, kernel_size=3)

    # half-wave rectify
    spec_flux = np.maximum(spec_flux, 0)
    
    # Normalize spec_flux
    spec_flux /= max(spec_flux)

    return time_frames[1:], spec_flux # skip the first in time_frames due to difference calculation


def detect_onsets_and_release(spec_flux, times, sr, onset_thresh=0.15, sustain_thresh=.13, min_time_between=0.15):
    onsets = [] # indices onsets incur
    sustains = [] # indices sustains incur
    search_sustain: bool = False # if true, then we're looking for sustain phase. if false, we're looking for next onset
    onset_init: int = 0

    mtb_samples = int(min_time_between * sr * .2)

    for i in range(1, len(spec_flux) - 1):
        # keep track of last time we we're below threshold -> will be when not articulation begins
        onset_init = i if spec_flux[i] < onset_thresh else onset_init

        if not search_sustain:
            if not onsets or (times[i] - times[onsets[-1]]) > min_time_between:
                if (spec_flux[i] > spec_flux[i-1] and spec_flux[i] >= spec_flux[i+1] and spec_flux[i] > onset_thresh):
                    onsets.append(onset_init)
                    search_sustain = True
        else:
            window = np.arange(i, min(i + mtb_samples, len(times)))
            # window = np.arange(i, min(onsets[-1] + mtb_samples, len(times)))
            if np.average(spec_flux[window]) < sustain_thresh:
                sustains.append(i)
                search_sustain = False
    onsets = [times[i] for i in onsets]
    sustains = [times[i] for i in sustains]

    return onsets, sustains

# hop_len is the hop_len for librosa.pyin
def detect_onsets_with_f0(spec_flux, times, sr, f0, voiced, hop_len, threshold=0.15, min_time_between=0.15, min_hz_diff=20):
    onsets = []

    seconds_per_f0_read = hop_len / sr
    
    # number of f0 reads for chunk of time over min_time_between
    num_f0s = min_time_between / seconds_per_f0_read

    window_size = max(1, int(num_f0s))
    
    # Find local maxima above threshold 
    for i in range(1, len(spec_flux)-1): 
        # Check timing with previous onset
        if not onsets or (times[i] - onsets[-1]) > min_time_between:
            # Check if it's a local peak above threshold 
            flux_peak = (spec_flux[i] > spec_flux[i-1] and spec_flux[i] >= spec_flux[i+1] and spec_flux[i] > threshold)

            if flux_peak:  # ### CHANGED: early-accept flux peaks
                onsets.append(times[i])
                continue

            # check for note fundamental frequency change (f0)
            f0_frame = int(i / hop_len)

            window_left = np.arange(max(0, f0_frame - window_size), max(0, f0_frame - 1))
            window_right = np.arange(f0_frame, min(len(f0) - 1, f0_frame + window_size))

            left_vals_raw  = f0[window_left]
            right_vals_raw = f0[window_right]

            left_mask  = voiced[window_left]  & np.isfinite(left_vals_raw)  & (left_vals_raw  > 0)
            right_mask = voiced[window_right] & np.isfinite(right_vals_raw) & (right_vals_raw > 0)

            left_vals  = left_vals_raw[left_mask]
            right_vals = right_vals_raw[right_mask]

            # check that all f0 on right and left window are sufficiently different
            # lr_diff = [[abs(f0[right] - f0[left]) >= min_hz_diff for left in window_left] for right in window_right]
            # lf_window_diff: bool = all([all(row) for row in lr_diff])

            # check that all values in the right window are sufficiently similar
            r_spread = (np.max(right_vals) - np.min(right_vals)) if len(right_vals) > 1 else 0.0
            all_r_same = r_spread < min_hz_diff  # ### CHANGED: use filtered arrays and robust spread

            # Compare medians across windows (instead of all-pairs, which was too strict)
            l_med = float(np.median(left_vals))
            r_med = float(np.median(right_vals))
            hz_jump = abs(r_med - l_med) 

            f0_diff = voiced[f0_frame] and hz_jump >= min_hz_diff and all_r_same

            if f0_diff:
                onsets.append(times[i])
    return onsets

def extract_notes_from_recording(mp4_file, min_note_duration=0.1):
    ys, ts, sr = get_audio_data(mp4_file)
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=ys, 
        sr=sr, 
        fmin=librosa.note_to_hz('E3'), 
        fmax=librosa.note_to_hz('C6'),
        frame_length=2048,
        hop_length=512
    )
    
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
    
    notes = []
    current_note = None
    min_frames = int(min_note_duration / (512/sr)) 
    
    for i in range(len(f0)):
        if voiced_flag[i] and not np.isnan(f0[i]):
            note_name = librosa.hz_to_note(f0[i], octave=True)
            
            if current_note is None or current_note["name"] != note_name:
                if current_note and (i - current_note["start_frame"]) >= min_frames:
                    notes.append({
                        "name": current_note["name"],
                        "start_time": frame_times[current_note["start_frame"]],
                        "end_time": frame_times[i-1],
                        "duration": frame_times[i-1] - frame_times[current_note["start_frame"]],
                        "avg_freq": np.mean(current_note["freqs"])
                    })
                
                current_note = {
                    "name": note_name,
                    "start_frame": i,
                    "freqs": [f0[i]]
                }
            else:
                current_note["freqs"].append(f0[i])
    
    if current_note and (len(f0) - current_note["start_frame"]) >= min_frames:
        notes.append({
            "name": current_note["name"],
            "start_time": frame_times[current_note["start_frame"]],
            "end_time": frame_times[-1],
            "duration": frame_times[-1] - frame_times[current_note["start_frame"]],
            "avg_freq": np.mean(current_note["freqs"])
        })
    
    print(f"Detected {len(notes)} notes in {mp4_file}:")
    for i, note in enumerate(notes):
        print(f"{i+1}. {note['name']} ({note['start_time']:.2f}s - {note['end_time']:.2f}s, {note['duration']:.2f}s)")
    
    return notes

# notes = extract_notes_from_recording("ex1WholeModF.mp4")
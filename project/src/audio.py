"""Five-second mono audio -> 64-band log-mel spectrogram."""
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.runtime').exists():
    sys.path.insert(0, str(ROOT / '.runtime'))
import soundfile as sf

SAMPLE_RATE = 22050
SECONDS = 5
N_FFT = 1024
HOP = 512
N_MELS = 64


def mel_filter():
    # HTK mel scale: equal steps in mel give finer bands at low frequencies.
    mel_max = 2595 * np.log10(1 + SAMPLE_RATE / 2 / 700)
    frequencies = 700 * (10 ** (np.linspace(0, mel_max, N_MELS + 2) / 2595) - 1)
    fft_frequencies = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    weights = np.zeros((N_MELS, len(fft_frequencies)), dtype=np.float32)
    for band in range(N_MELS):
        left, center, right = frequencies[band:band + 3]
        weights[band] = np.maximum(0, np.minimum(
            (fft_frequencies - left) / (center - left),
            (right - fft_frequencies) / (right - center)))
    return torch.from_numpy(weights)


MEL_FILTER = mel_filter()
WINDOW = torch.hann_window(N_FFT)


def load_segment(path):
    # Use the middle five seconds consistently in training and prediction.
    with sf.SoundFile(str(path)) as audio:
        sample_rate = audio.samplerate
        frames = SECONDS * sample_rate
        audio.seek(max(0, (len(audio) - frames) // 2))
        signal = audio.read(frames, dtype='float32', always_2d=True).mean(axis=1)
    if len(signal) == 0 or not np.isfinite(signal).all():
        raise ValueError('The file has no valid audio samples.')
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // divisor, sample_rate // divisor)
    length = SAMPLE_RATE * SECONDS
    signal = np.pad(signal[:length], (0, max(0, length - len(signal))))
    return torch.from_numpy(signal.astype(np.float32))


def spectrogram(signal):
    spectrum = torch.stft(signal, n_fft=N_FFT, hop_length=HOP, window=WINDOW,
                          return_complex=True).abs().square()
    return torch.log10((MEL_FILTER @ spectrum).clamp_min(1e-10)).unsqueeze(0)


def prepare(path):
    start = perf_counter()
    signal = load_segment(path)
    decoded = perf_counter()
    feature = spectrogram(signal)
    end = perf_counter()
    return feature, {'audio_loading_ms': 1000 * (decoded - start),
                     'spectrogram_ms': 1000 * (end - decoded)}

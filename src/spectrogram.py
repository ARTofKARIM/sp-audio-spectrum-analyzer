"""Spectrogram computation module."""

import numpy as np
from src.fft_processor import FFTProcessor


class SpectrogramGenerator:
    """Generates spectrograms from audio signals."""

    def __init__(self, fft_size=2048, hop_size=512, window_type="hann"):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.fft = FFTProcessor(fft_size, window_type)

    def compute_stft(self, signal):
        """Compute Short-Time Fourier Transform."""
        n_frames = 1 + (len(signal) - self.fft_size) // self.hop_size
        n_bins = self.fft_size // 2 + 1
        stft_matrix = np.zeros((n_bins, n_frames), dtype=complex)

        for i in range(n_frames):
            start = i * self.hop_size
            frame = signal[start:start + self.fft_size]
            stft_matrix[:, i] = self.fft.compute_fft(frame)
        return stft_matrix

    def magnitude_spectrogram(self, signal):
        """Compute magnitude spectrogram."""
        stft = self.compute_stft(signal)
        return np.abs(stft)

    def power_spectrogram(self, signal):
        """Compute power spectrogram."""
        stft = self.compute_stft(signal)
        return np.abs(stft) ** 2

    def to_db(self, spectrogram, ref=1.0, top_db=80.0):
        """Convert spectrogram to decibel scale."""
        db = 20.0 * np.log10(spectrogram / ref + 1e-10)
        db = np.maximum(db, db.max() - top_db)
        return db

    def mel_spectrogram(self, signal, sample_rate, n_mels=128, fmin=20, fmax=8000):
        """Compute mel-scaled spectrogram."""
        power_spec = self.power_spectrogram(signal)
        mel_basis = self._mel_filterbank(sample_rate, n_mels, fmin, fmax)
        mel_spec = np.dot(mel_basis, power_spec)
        return mel_spec

    def _mel_filterbank(self, sample_rate, n_mels, fmin, fmax):
        """Create mel filterbank matrix."""
        n_bins = self.fft_size // 2 + 1
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bins = np.floor((self.fft_size + 1) * hz_points / sample_rate).astype(int)

        filterbank = np.zeros((n_mels, n_bins))
        for i in range(n_mels):
            for j in range(bins[i], bins[i + 1]):
                if j < n_bins:
                    filterbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i] + 1e-8)
            for j in range(bins[i + 1], bins[i + 2]):
                if j < n_bins:
                    filterbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1] + 1e-8)
        return filterbank

    def time_axis(self, signal_length, sample_rate):
        n_frames = 1 + (signal_length - self.fft_size) // self.hop_size
        return np.arange(n_frames) * self.hop_size / sample_rate

    def frequency_axis(self, sample_rate):
        return np.fft.rfftfreq(self.fft_size, d=1.0 / sample_rate)

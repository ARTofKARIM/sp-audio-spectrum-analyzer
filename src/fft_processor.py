"""FFT processing module with configurable parameters."""

import numpy as np
from src.windowing import WindowFunction


class FFTProcessor:
    """Performs FFT analysis on audio signals."""

    def __init__(self, fft_size=2048, window_type="hann"):
        self.fft_size = fft_size
        self.window = WindowFunction.get(window_type, fft_size)
        self.window_type = window_type

    def compute_fft(self, signal):
        """Compute the FFT of a signal frame."""
        if len(signal) < self.fft_size:
            signal = np.pad(signal, (0, self.fft_size - len(signal)))
        elif len(signal) > self.fft_size:
            signal = signal[:self.fft_size]
        windowed = signal * self.window
        spectrum = np.fft.rfft(windowed, n=self.fft_size)
        return spectrum

    def magnitude_spectrum(self, signal):
        """Compute magnitude spectrum in linear scale."""
        spectrum = self.compute_fft(signal)
        return np.abs(spectrum)

    def power_spectral_density(self, signal):
        """Compute power spectral density."""
        spectrum = self.compute_fft(signal)
        return np.abs(spectrum) ** 2 / self.fft_size

    def magnitude_db(self, signal, ref=1.0):
        """Compute magnitude spectrum in decibels."""
        magnitude = self.magnitude_spectrum(signal)
        return 20.0 * np.log10(magnitude / ref + 1e-10)

    def frequency_axis(self, sample_rate):
        """Generate the frequency axis for the FFT output."""
        return np.fft.rfftfreq(self.fft_size, d=1.0 / sample_rate)

    def compute_batch(self, frames, sample_rate):
        """Compute FFT for a batch of frames."""
        freqs = self.frequency_axis(sample_rate)
        magnitudes = np.array([self.magnitude_db(frame) for frame in frames])
        return freqs, magnitudes

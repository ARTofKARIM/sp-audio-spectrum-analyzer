"""Frequency analysis tools for audio signals."""

import numpy as np
from scipy.signal import find_peaks


class FrequencyAnalyzer:
    """Advanced frequency domain analysis."""

    def __init__(self, sample_rate, fft_size=2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size

    def detect_peaks(self, magnitude_spectrum, height_db=-40, distance=10):
        """Find spectral peaks above a threshold."""
        peaks, properties = find_peaks(magnitude_spectrum, height=height_db, distance=distance)
        freqs = peaks * self.sample_rate / self.fft_size
        return freqs, magnitude_spectrum[peaks]

    def fundamental_frequency(self, signal):
        """Estimate fundamental frequency using autocorrelation."""
        signal = signal - np.mean(signal)
        corr = np.correlate(signal, signal, mode="full")
        corr = corr[len(corr) // 2:]
        corr = corr / corr[0]

        min_lag = int(self.sample_rate / 2000)  # 2000 Hz max
        max_lag = int(self.sample_rate / 50)    # 50 Hz min

        search = corr[min_lag:max_lag]
        if len(search) == 0:
            return 0.0
        peak = np.argmax(search) + min_lag
        return self.sample_rate / peak if peak > 0 else 0.0

    def harmonic_analysis(self, magnitude_spectrum, fundamental_freq, n_harmonics=5):
        """Analyze harmonic content relative to fundamental."""
        freq_resolution = self.sample_rate / self.fft_size
        harmonics = {}
        for h in range(1, n_harmonics + 1):
            target_freq = fundamental_freq * h
            bin_idx = int(target_freq / freq_resolution)
            if bin_idx < len(magnitude_spectrum):
                harmonics[f"H{h}"] = {"frequency": target_freq, "magnitude": float(magnitude_spectrum[bin_idx])}
        return harmonics

    def spectral_centroid(self, magnitude_spectrum):
        """Compute spectral centroid (center of mass of spectrum)."""
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        return np.sum(freqs * magnitude_spectrum) / (np.sum(magnitude_spectrum) + 1e-8)

    def spectral_rolloff(self, magnitude_spectrum, threshold=0.85):
        """Frequency below which threshold% of spectral energy is contained."""
        total_energy = np.sum(magnitude_spectrum)
        cumulative = np.cumsum(magnitude_spectrum)
        rolloff_idx = np.searchsorted(cumulative, threshold * total_energy)
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        return freqs[min(rolloff_idx, len(freqs) - 1)]

    def bandwidth(self, magnitude_spectrum, centroid=None):
        """Compute spectral bandwidth around centroid."""
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        if centroid is None:
            centroid = self.spectral_centroid(magnitude_spectrum)
        deviation = np.abs(freqs - centroid)
        return np.sum(deviation * magnitude_spectrum) / (np.sum(magnitude_spectrum) + 1e-8)

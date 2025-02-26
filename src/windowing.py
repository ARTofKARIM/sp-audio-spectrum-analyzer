"""Window function generation for spectral analysis."""

import numpy as np


class WindowFunction:
    """Provides various window functions for FFT analysis."""

    @staticmethod
    def hamming(size):
        return np.hamming(size)

    @staticmethod
    def hann(size):
        return np.hanning(size)

    @staticmethod
    def blackman(size):
        return np.blackman(size)

    @staticmethod
    def kaiser(size, beta=14.0):
        return np.kaiser(size, beta)

    @staticmethod
    def bartlett(size):
        return np.bartlett(size)

    @staticmethod
    def rectangular(size):
        return np.ones(size)

    @classmethod
    def get(cls, name, size, **kwargs):
        """Get a window function by name."""
        windows = {
            "hamming": cls.hamming,
            "hann": cls.hann,
            "blackman": cls.blackman,
            "kaiser": cls.kaiser,
            "bartlett": cls.bartlett,
            "rectangular": cls.rectangular,
        }
        if name not in windows:
            raise ValueError(f"Unknown window: {name}. Available: {list(windows.keys())}")
        return windows[name](size, **kwargs)

    @staticmethod
    def main_lobe_width(window_type, fft_size):
        """Approximate main lobe width in bins."""
        widths = {"rectangular": 2, "hann": 4, "hamming": 4, "blackman": 6, "kaiser": 5}
        return widths.get(window_type, 4)

    @staticmethod
    def sidelobe_level(window_type):
        """Approximate peak sidelobe level in dB."""
        levels = {"rectangular": -13, "hann": -32, "hamming": -43, "blackman": -58, "kaiser": -60}
        return levels.get(window_type, -30)

    @classmethod
    def list_windows(cls):
        return ["hamming", "hann", "blackman", "kaiser", "bartlett", "rectangular"]

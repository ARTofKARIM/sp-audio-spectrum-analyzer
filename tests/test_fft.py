"""Unit tests for FFT processing."""

import unittest
import numpy as np
from src.fft_processor import FFTProcessor
from src.windowing import WindowFunction


class TestFFTProcessor(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 44100
        self.fft_size = 2048
        self.fft = FFTProcessor(self.fft_size, "hann")

    def test_sine_wave_peak(self):
        freq = 1000
        t = np.arange(self.fft_size) / self.sample_rate
        signal = np.sin(2 * np.pi * freq * t)
        magnitude = self.fft.magnitude_spectrum(signal)
        freqs = self.fft.frequency_axis(self.sample_rate)
        peak_freq = freqs[np.argmax(magnitude)]
        self.assertAlmostEqual(peak_freq, freq, delta=50)

    def test_fft_output_size(self):
        signal = np.random.randn(self.fft_size)
        spectrum = self.fft.compute_fft(signal)
        self.assertEqual(len(spectrum), self.fft_size // 2 + 1)

    def test_magnitude_non_negative(self):
        signal = np.random.randn(self.fft_size)
        magnitude = self.fft.magnitude_spectrum(signal)
        self.assertTrue(np.all(magnitude >= 0))


class TestWindowing(unittest.TestCase):

    def test_window_length(self):
        for name in WindowFunction.list_windows():
            w = WindowFunction.get(name, 1024)
            self.assertEqual(len(w), 1024)

    def test_hann_endpoints(self):
        w = WindowFunction.hann(256)
        self.assertAlmostEqual(w[0], 0.0, places=5)

    def test_rectangular_all_ones(self):
        w = WindowFunction.rectangular(100)
        self.assertTrue(np.allclose(w, 1.0))


if __name__ == "__main__":
    unittest.main()

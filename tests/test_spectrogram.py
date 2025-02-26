"""Unit tests for spectrogram module."""

import unittest
import numpy as np
from src.spectrogram import SpectrogramGenerator


class TestSpectrogram(unittest.TestCase):

    def setUp(self):
        self.sr = 44100
        self.duration = 1.0
        self.signal = np.random.randn(int(self.sr * self.duration)).astype(np.float32)
        self.spec_gen = SpectrogramGenerator(fft_size=1024, hop_size=256)

    def test_spectrogram_shape(self):
        spec = self.spec_gen.magnitude_spectrogram(self.signal)
        n_bins = 1024 // 2 + 1
        self.assertEqual(spec.shape[0], n_bins)

    def test_db_conversion(self):
        spec = self.spec_gen.magnitude_spectrogram(self.signal)
        db = self.spec_gen.to_db(spec)
        self.assertTrue(np.all(np.isfinite(db)))

    def test_chirp_spectrogram(self):
        t = np.linspace(0, 1, self.sr)
        chirp = np.sin(2 * np.pi * np.cumsum(np.linspace(100, 4000, self.sr)) / self.sr)
        spec = self.spec_gen.magnitude_spectrogram(chirp.astype(np.float32))
        self.assertGreater(spec.shape[1], 0)

    def test_time_axis(self):
        t = self.spec_gen.time_axis(len(self.signal), self.sr)
        self.assertAlmostEqual(t[-1], self.duration, delta=0.1)


if __name__ == "__main__":
    unittest.main()

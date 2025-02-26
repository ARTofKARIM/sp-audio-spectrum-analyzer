"""Audio input/output module for reading and writing audio files."""

import numpy as np
import soundfile as sf
import os


class AudioReader:
    """Reads audio files and provides metadata."""

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None
        self.sample_rate = None
        self.duration = None
        self.channels = None

    def load(self, filepath=None):
        """Load audio from a WAV file."""
        path = filepath or self.filepath
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        self.data, self.sample_rate = sf.read(path, dtype="float32")
        if self.data.ndim > 1:
            self.channels = self.data.shape[1]
        else:
            self.channels = 1
        self.duration = len(self.data) / self.sample_rate
        self.filepath = path
        print(f"Loaded: {path}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Duration: {self.duration:.2f} s")
        print(f"  Channels: {self.channels}")
        return self.data

    def to_mono(self):
        """Convert stereo to mono by averaging channels."""
        if self.data is None:
            raise ValueError("No audio loaded")
        if self.data.ndim > 1:
            self.data = np.mean(self.data, axis=1)
            self.channels = 1
        return self.data

    def get_segment(self, start_time, end_time):
        """Extract a time segment from the audio."""
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        return self.data[start_sample:end_sample]

    def get_frames(self, frame_size, hop_size):
        """Split audio into overlapping frames."""
        if self.data is None:
            raise ValueError("No audio loaded")
        signal = self.data if self.data.ndim == 1 else self.data[:, 0]
        n_frames = 1 + (len(signal) - frame_size) // hop_size
        frames = np.zeros((n_frames, frame_size))
        for i in range(n_frames):
            start = i * hop_size
            frames[i] = signal[start:start + frame_size]
        return frames


class AudioWriter:
    """Writes audio data to files."""

    @staticmethod
    def save(filepath, data, sample_rate):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        sf.write(filepath, data, sample_rate)
        print(f"Saved: {filepath}")

    @staticmethod
    def generate_sine(frequency, duration, sample_rate=44100, amplitude=0.5):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return amplitude * np.sin(2 * np.pi * frequency * t)

    @staticmethod
    def generate_chirp(f_start, f_end, duration, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        freq = np.linspace(f_start, f_end, len(t))
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        return np.sin(phase).astype(np.float32)

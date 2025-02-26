"""Visualization module for audio spectrum analysis."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SpectrumVisualizer:
    """Generates plots for audio spectrum analysis."""

    def __init__(self, output_dir="output/", figsize=(12, 6), dpi=150):
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi

    def plot_waveform(self, signal, sample_rate, title="Waveform", save=True):
        t = np.arange(len(signal)) / sample_rate
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(t, signal, linewidth=0.5, color="steelblue")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        if save:
            fig.savefig(f"{self.output_dir}waveform.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_magnitude_spectrum(self, freqs, magnitude_db, title="Magnitude Spectrum", save=True):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(freqs, magnitude_db, linewidth=0.8, color="steelblue")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(title)
        ax.set_xlim(0, freqs[-1])
        ax.grid(True, alpha=0.3)
        if save:
            fig.savefig(f"{self.output_dir}spectrum.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_spectrogram(self, spectrogram_db, time_axis, freq_axis, title="Spectrogram",
                         cmap="viridis", save=True):
        fig, ax = plt.subplots(figsize=self.figsize)
        img = ax.pcolormesh(time_axis, freq_axis, spectrogram_db, shading="auto", cmap=cmap)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        plt.colorbar(img, ax=ax, label="dB")
        if save:
            fig.savefig(f"{self.output_dir}spectrogram.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_mel_spectrogram(self, mel_spec_db, time_axis, title="Mel Spectrogram", save=True):
        fig, ax = plt.subplots(figsize=self.figsize)
        img = ax.imshow(mel_spec_db, aspect="auto", origin="lower", cmap="magma",
                        extent=[time_axis[0], time_axis[-1], 0, mel_spec_db.shape[0]])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel Band")
        ax.set_title(title)
        plt.colorbar(img, ax=ax, label="dB")
        if save:
            fig.savefig(f"{self.output_dir}mel_spectrogram.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_waterfall_3d(self, spectrogram_db, time_axis, freq_axis, save=True):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection="3d")
        T, F = np.meshgrid(time_axis, freq_axis)
        ax.plot_surface(T, F, spectrogram_db, cmap="viridis", alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_zlabel("dB")
        ax.set_title("3D Waterfall Spectrogram")
        if save:
            fig.savefig(f"{self.output_dir}waterfall_3d.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_window_comparison(self, fft_size=2048, save=True):
        from src.windowing import WindowFunction
        windows = ["rectangular", "hann", "hamming", "blackman"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        for name in windows:
            w = WindowFunction.get(name, fft_size)
            ax1.plot(w, label=name, linewidth=1.5)
            W = np.fft.rfft(w, n=fft_size * 4)
            mag_db = 20 * np.log10(np.abs(W) / np.max(np.abs(W)) + 1e-10)
            ax2.plot(mag_db[:500], label=name, linewidth=1.5)
        ax1.set_title("Window Functions (Time Domain)")
        ax1.legend()
        ax2.set_title("Window Functions (Frequency Response)")
        ax2.set_ylabel("dB")
        ax2.set_ylim(-100, 5)
        ax2.legend()
        if save:
            fig.savefig(f"{self.output_dir}windows.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

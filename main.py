"""CLI entry point for the audio spectrum analyzer."""

import argparse
import yaml
import numpy as np
from src.audio_io import AudioReader, AudioWriter
from src.fft_processor import FFTProcessor
from src.spectrogram import SpectrogramGenerator
from src.frequency_analysis import FrequencyAnalyzer
from src.visualization import SpectrumVisualizer


def main():
    parser = argparse.ArgumentParser(description="Audio Spectrum Analyzer")
    parser.add_argument("input", nargs="?", help="Path to audio file")
    parser.add_argument("--generate", choices=["sine", "chirp"], help="Generate test signal")
    parser.add_argument("--freq", type=float, default=440.0, help="Frequency for sine generation")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    parser.add_argument("--analysis", choices=["fft", "spectrogram", "all"], default="all")
    args = parser.parse_args()

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    viz = SpectrumVisualizer()

    if args.generate:
        sr = config["audio"]["sample_rate"]
        if args.generate == "sine":
            signal = AudioWriter.generate_sine(args.freq, args.duration, sr)
        else:
            signal = AudioWriter.generate_chirp(200, 2000, args.duration, sr)
        AudioWriter.save("audio/generated.wav", signal, sr)
    else:
        reader = AudioReader()
        signal = reader.load(args.input)
        signal = reader.to_mono()
        sr = reader.sample_rate

    fft_cfg = config["fft"]

    if args.analysis in ["fft", "all"]:
        fft = FFTProcessor(fft_cfg["fft_size"], fft_cfg["window"])
        freqs = fft.frequency_axis(sr)
        mag_db = fft.magnitude_db(signal[:fft_cfg["fft_size"]])
        viz.plot_waveform(signal, sr)
        viz.plot_magnitude_spectrum(freqs, mag_db)

        analyzer = FrequencyAnalyzer(sr, fft_cfg["fft_size"])
        f0 = analyzer.fundamental_frequency(signal[:sr])
        print(f"Estimated fundamental frequency: {f0:.1f} Hz")

    if args.analysis in ["spectrogram", "all"]:
        spec_gen = SpectrogramGenerator(fft_cfg["fft_size"], fft_cfg["hop_size"], fft_cfg["window"])
        spec = spec_gen.magnitude_spectrogram(signal)
        spec_db = spec_gen.to_db(spec)
        t_axis = spec_gen.time_axis(len(signal), sr)
        f_axis = spec_gen.frequency_axis(sr)
        viz.plot_spectrogram(spec_db, t_axis, f_axis)

    print("Analysis complete. Results saved to output/")


if __name__ == "__main__":
    main()

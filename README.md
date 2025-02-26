# Audio Spectrum Analyzer

A signal processing toolkit for audio frequency analysis using FFT. Supports WAV file analysis, spectrogram generation, mel-spectrograms, and advanced frequency analysis with configurable windowing functions.

## Overview

This project implements a complete audio spectrum analysis pipeline covering FFT computation, spectrogram generation, frequency peak detection, and fundamental frequency estimation. It provides both a CLI interface and a modular Python API.

## Architecture

```
sp-audio-spectrum-analyzer/
├── src/
│   ├── audio_io.py            # WAV file reading/writing, signal generation
│   ├── fft_processor.py       # FFT computation with windowing
│   ├── windowing.py           # Window functions (Hann, Hamming, Blackman, etc.)
│   ├── spectrogram.py         # STFT, mel-spectrogram generation
│   ├── frequency_analysis.py  # Peak detection, F0 estimation, spectral features
│   └── visualization.py       # Waveform, spectrum, spectrogram, 3D plots
├── config/config.yaml
├── tests/
│   ├── test_fft.py
│   └── test_spectrogram.py
└── main.py
```

## Features

- **FFT Analysis**: Configurable FFT size, multiple window functions
- **Spectrogram**: STFT-based spectrogram with dB scaling
- **Mel Spectrogram**: Perceptually-weighted frequency representation
- **Frequency Analysis**: Peak detection, harmonics, spectral centroid, bandwidth, rolloff
- **Signal Generation**: Sine waves, chirp signals for testing
- **Visualization**: Waveform, spectrum, spectrogram, 3D waterfall plots

## Window Functions

| Window | Main Lobe | Sidelobe Level | Best For |
|--------|-----------|---------------|----------|
| Rectangular | Narrowest | -13 dB | Frequency resolution |
| Hann | Medium | -32 dB | General purpose |
| Hamming | Medium | -43 dB | Speech analysis |
| Blackman | Widest | -58 dB | Dynamic range |

## Installation

```bash
git clone https://github.com/mouachiqab/sp-audio-spectrum-analyzer.git
cd sp-audio-spectrum-analyzer
pip install -r requirements.txt
```

## Usage

```bash
# Analyze a WAV file
python main.py audio/sample.wav --analysis all

# Generate and analyze a test signal
python main.py --generate sine --freq 440 --duration 2.0

# Spectrogram only
python main.py audio/sample.wav --analysis spectrogram
```

## Technologies

- Python 3.9+
- NumPy, SciPy (FFT, signal processing)
- Matplotlib (visualization)
- soundfile (audio I/O)





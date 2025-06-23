import torch
import numpy as np
from demucs.spec import spectro

def spectroAPI(x, n_fft=512, hop_length=None, pad=0):
    return NotImplementedError("This function is just a placeholder for the spectro function.")

def test_spectrogram_shape_pytorch():
    """ Test the shape of the spectrogram output using PyTorch's STFT """
    batch, channels, samples = 1, 2, 2*44100
    n_fft, hop_length = 4096, 1024
    waveform = torch.randn(batch, channels, samples)  # (batch, channels, samples)
    spec = spectro(waveform, n_fft=n_fft, hop_length=hop_length, pad=0, stft_type='pytorch')
    assert spec.shape == (1, 2, n_fft // 2 + 1, samples // hop_length + 1, 2)  # (batch, channels, freq_bins, time_steps, 2)

def test_spectrogram_shape_custom():
    """ Test the shape of the spectrogram output using Custom STFT """
    batch, channels, samples = 1, 2, 20*44100
    n_fft, hop_length = 4096, 1024
    waveform = torch.randn(batch, channels, samples)  # (batch, channels, samples)
    spec = spectro(waveform, n_fft=n_fft, hop_length=hop_length, pad=0, stft_type='custom')
    assert spec.shape == (1, 2, n_fft // 2 + 1, samples // hop_length + 1, 2)  # (batch, channels, freq_bins, time_steps, 2)

def test_compare_spectrograms():
    """ Compare the spectrograms from PyTorch and Custom STFT """
    batch, channels, samples = 1, 2, 20*44100
    n_fft, hop_length = 4096, 1024
    waveform = torch.randn(batch, channels, samples)  # (batch, channels, samples)

    spec_pytorch = spectro(waveform, n_fft=n_fft, hop_length=hop_length, pad=0, stft_type='pytorch')
    spec_custom = spectro(waveform, n_fft=n_fft, hop_length=hop_length, pad=0, stft_type='custom')

    # Calculate the mean difference between the two spectrograms
    mean_diff = torch.abs(spec_pytorch - spec_custom).mean().item()
    print("\nSTFT Result: Mean Difference =", mean_diff)

    assert torch.allclose(spec_pytorch, spec_custom, atol=1e-3), "Spectrograms do not match!"
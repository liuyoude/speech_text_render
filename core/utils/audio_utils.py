# -*- coding: utf-8 -*-
"""
Shared audio processing utilities.
"""
import librosa
import numpy as np
import torch


def calculate_perceptual_energy(y, sr, frame_length, hop_length):
    """A-weighted perceptual energy per frame.

    Computes the STFT magnitude, applies an A-weighting curve to
    approximate human loudness perception, and returns the RMS-like
    energy normalised by *frame_length*.

    Parameters
    ----------
    y : np.ndarray
        Audio time-series (mono).
    sr : int
        Sample rate of *y*.
    frame_length : int
        FFT window size (samples, should be a power of 2).
    hop_length : int
        Hop size between frames (samples).

    Returns
    -------
    np.ndarray
        1-D array of per-frame perceptual energy values.
    """
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    a_weighting = librosa.A_weighting(freqs + 1e-8)
    a_weighting = librosa.db_to_power(a_weighting)
    a_weighting = np.expand_dims(a_weighting, axis=1)
    S = S * a_weighting

    perceptual_energy = np.sqrt(np.sum(S**2, axis=0)) / frame_length
    return perceptual_energy


_fcpe_model = None
_fcpe_device = None


def _get_fcpe_model(device="cpu"):
    """Lazily load the bundled FCPE model (singleton)."""
    global _fcpe_model, _fcpe_device
    if _fcpe_model is None or _fcpe_device != str(device):
        from torchfcpe import spawn_bundled_infer_model
        _fcpe_model = spawn_bundled_infer_model(device=device)
        _fcpe_device = str(device)
    return _fcpe_model


def extract_f0(y, sr, hop_ms=10, f0_min=50.0, f0_max=1100.0, device="cpu"):
    """Extract per-frame F0 using FCPE (torchfcpe).

    Parameters
    ----------
    y : np.ndarray
        Audio time-series (mono, 1-D).
    sr : int
        Sample rate of *y*.
    hop_ms : float
        Hop size in milliseconds. Determines output frame rate.
    f0_min : float
        Minimum F0 in Hz.
    f0_max : float
        Maximum F0 in Hz.
    device : str
        Torch device string (e.g. ``"cpu"``, ``"cuda"``).

    Returns
    -------
    f0 : np.ndarray
        1-D array of F0 values in Hz. Unvoiced frames are 0.
    hop_length : int
        Hop size in samples (for time alignment with energy frames).
    """
    model = _get_fcpe_model(device)
    wav_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(-1)
    wav_tensor = wav_tensor.to(device)

    hop_length = int(sr * hop_ms / 1000)
    target_length = y.shape[0] // hop_length + 1

    with torch.no_grad():
        f0_tensor = model.infer(
            wav_tensor, sr,
            decoder_mode="local_argmax",
            threshold=0.006,
            f0_min=f0_min,
            f0_max=f0_max,
            interp_uv=False,
            output_interp_target_length=target_length,
        )
    f0 = f0_tensor.squeeze().cpu().numpy()
    return f0, hop_length

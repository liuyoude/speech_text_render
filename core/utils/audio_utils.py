# -*- coding: utf-8 -*-
"""
Shared audio processing utilities.
"""
import librosa
import numpy as np


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

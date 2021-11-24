import pandas as pd
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from util import smoothing

def fourier_transform(arr: pd.DataFrame, ws: int) -> tuple:
    """
    apply the high-pass filter and low-pass filter
    high-pass filter: only high frequency could pass, used to extract noise and outliers
    low-pass filter: only low frequency could pass, used to remove noise and outliers
    @param arr: whole time series 
    @param ws: window size
    """
    fft_orig = fft(arr["orig"].values) 

    # sample frequency in d time step
    sample_freq = fftfreq(len(arr), d = 1 / ws)
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    power = np.abs(fft_orig)**2
    peak_freq = freqs[power[pos_mask].argmax()]

    # apply low-pass filter, compute the residuals
    low_freq_fft = fft_orig.copy()
    low_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    low_fft_res = np.abs(arr["orig"] - ifft(low_freq_fft))

    # apply high-pass filter
    high_freq_fft = fft_orig.copy()
    high_freq_fft[np.abs(sample_freq) < peak_freq] = 0
    high_fft = pd.Series(np.abs(ifft(high_freq_fft)))

    # apply smoothing
    low_fft_score  = smoothing(low_fft_res.values, ws)
    high_fft_score = smoothing(high_fft.values, ws)

    return low_fft_score, high_fft_score

import glob
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft, fftfreq
import stumpy
import matplotlib.pyplot as plt


def fourier_transform(arr: pd.DataFrame, ws: int, split_point: int) -> tuple:
    """
    apply the high-pass filter and low-pass filter
    high-pass filter: only high frequency could pass, used to extract noise and outliers
    low-pass filter: only low frequency could pass, used to remove noise and outliers
    @param arr: whole time series 
    @param ws: window size
    @param split_point: point to split the train and test set
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

def smoothing(arr: pd.Series, ws: int, padding_len=3) -> pd.Series:
    """
    @param arr: a pandas series
    @param ws: window size
    @param padding_len: default to be 3
    return the smoothing scores
    """
    padding = ws * padding_len
    arr = pd.DataFrame(arr, columns=["metric"])

    arr['mask'] = 0.0
    arr.loc[arr.index[ws:-ws-padding], 'mask'] = 1.0
    arr['mask'] = arr['mask'].rolling(padding, min_periods=1).sum() / padding
    arr["score"] = arr["metric"].rolling(window=ws).mean() * arr['mask']
        
    return arr["score"]

def compute_confidence_score(s: pd.DataFrame, ws: int, split_point: int) -> tuple:
    """
    @param s: score dataframe
    @param ws: window _size
    @param split_point: index splitting train and test time series
    return the ratio of first peak and second peak
    """
    y = s.copy()
    
    # find local maximum
    local_max = (y == y.rolling(window=ws).max())
    y.loc[~local_max] = np.nan
    idx_1 = y.idxmax()
    peak_1 = y.max()
    
    # no local maximum
    if not np.isfinite(peak_1):
        # print("The value of the first peak is infinite")
        return None, None, None
    else:
        # the maximum is not in test set
        begin = idx_1 - ws
        end = idx_1 + ws
        if begin < split_point:
            # print("The first peak does not locate in the test set")
            return None, None, None
        else:
            # find peak_2 except for this interval [peak_1 - w, peak_1 + w)
            y.iloc[begin:end] = np.nan
            
            # idx_2 = y.idxmax()
            peak_2 = y.max()
            
            # if the second height is 0, skip
            if peak_2 == 0. or np.isnan(peak_2):
                # print("The second peak's height is 0 or nan")
                return None, None, None
            else:
                ratio = peak_1 / peak_2
                
    return ratio, idx_1, peak_1

def peak_to_peak_value(arr: pd.DataFrame, name: str, ws: int, split_point: int) -> tuple:
    """
    calculate the difference of rolling max and min in each window
    @param arr: dataframe of time series
    @param name: name of the time series
    @param ws: window size
    @param split_point: point to split the train and test set
    """
    slide_max = arr[name].rolling(window=ws).max()
    slide_min = arr[name].rolling(window=ws).min()
    score = (slide_max - slide_min).shift(-ws)
    smooth_score = smoothing(score.values, ws)
    conf, idx, peak = compute_confidence_score(smooth_score, ws, split_point)
    return conf, idx, peak, smooth_score

def inverse(arr: pd.Series, ws: int, split_point: int, threshold = 0.1) -> tuple:
    """
    clip any values below mean * threshold
    @param arr: dataframe of time series
    @param ws: window size
    @param split_point: point to split the train and test set
    @param threshold: clip the lower bound of the series
    """
    numerator = arr.mean()
    denominator = arr.clip(lower = numerator * threshold)
    score = numerator / denominator
    smooth_score = smoothing(score.values, ws)
    conf, idx, peak = compute_confidence_score(smooth_score, ws, split_point)
    return conf, idx, peak

def get_small_diff(arr: pd.DataFrame, name: str, ws: int, split_point: int, threshold=0.1) -> tuple:
    """
    region where the absolute value of the first order derivative is very small, less than the threshold quantile
    @param arr: time series
    @param name: name of the time series
    @param ws: window size
    @param split_point: point to split the train and test set
    @param threshold: threshold of quantile
    """
    arr_abs = arr[name].abs()
    cond = arr_abs <= arr_abs.quantile(threshold)
    score = cond.rolling(ws).mean().shift(-ws)
    smooth_score = smoothing(score.values, ws)
    conf, idx, peak = compute_confidence_score(smooth_score, ws, split_point)
    return conf, idx, peak

def get_std(arr: pd.DataFrame, name: str, ws: int, split_point: int) -> tuple:
    """
    standard deviation
    @param arr: time series
    @param name: name of the time series
    @param ws: window size
    @param split_point: point to split the train and test set
    """
    score = arr[name].rolling(window=ws).std().shift(-ws)
    smooth_score = smoothing(score.values, ws)
    conf, idx, peak = compute_confidence_score(smooth_score, ws, split_point)
    return conf, idx, peak, smooth_score

def orig_mp_outlier(arr: pd.DataFrame, ws: int, split_point: int) -> tuple:
    """
    use self-join version of matrix profile
    @param arr: time series, train + test
    @param ws: window size
    @param split_point: point to split the train and test set
    """
    # (distance, neighbor_idx, , )
    mp = stumpy.gpu_stump(arr["orig"], ws)
    mp_smooth = smoothing(mp[:, 0], ws)
    conf, idx, peak = compute_confidence_score(mp_smooth, ws, split_point)
    return conf, idx, peak

def orig_mp_novelty(train: pd.DataFrame, test: pd.DataFrame, arr: pd.DataFrame, ws: int, split_point: int) -> tuple:
    """
    for every subsequence in test, find its furthest subsequence in train set
    @param train: train time series
    @param test: test time series
    @param arr: full time series
    @param w: window size
    @param split_point: split point to separate train and test set
    """
    ab_mp = stumpy.gpu_stump(T_A = test["orig"], m = ws, T_B = train["orig"], ignore_trivial = False)
    begin = split_point
    end = begin + len(ab_mp) - 1
    arr.loc[begin:end, "mp_novelty"] = ab_mp[:,0]
    ab_mp_smooth = smoothing(arr["mp_novelty"].values, ws)
    conf, idx, peak = compute_confidence_score(ab_mp_smooth, ws, split_point)
    return conf, idx, peak
import pandas as pd
from util import smoothing, compute_confidence_score


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

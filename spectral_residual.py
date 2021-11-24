import sranodec as sr
import pandas as pd
from util import smoothing, compute_confidence_score

def spectral_residual(arr: pd.DataFrame, name: str, ws: int, split_point: int) -> tuple:
    """
    use spectral residual method to find the anomaly
    """
    # same as the period
    series_window_size = ws
    # less than period
    amp_window_size = int(series_window_size / 1.1)
    # a number enough larger than period
    score_window_size= 2 * series_window_size
    spec = sr.Silency(amp_window_size, series_window_size, score_window_size)
    score = spec.generate_anomaly_score(arr[name].values)
    score = smoothing(score, ws)
    conf, idx, peak = compute_confidence_score(score, ws, split_point)
    return conf, idx, peak
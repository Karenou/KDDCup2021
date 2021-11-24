import pandas as pd
import stumpy
from util import smoothing, compute_confidence_score

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
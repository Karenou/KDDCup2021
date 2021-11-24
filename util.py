import pandas as pd
import numpy as np

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
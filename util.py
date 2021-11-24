import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

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


def plot_anomaly(file_id: str, data: pd.DataFrame, result_df: pd.DataFrame, split_point: int) -> None:
    """
    plot the anomaly point against the whole time series data
    @param file_id: 
    @param data: whole time series
    @param result_df: store the confidence, idx of peak, and peak values
    @param split_point: point to split the train and test set
    """
    plt.figure(figsize=(20,4))
    plt.plot(np.arange(len(data)), data["orig"])

    anomaly_point = result_df["idx"][0]

    plt.axvline(x=split_point, ls=":", label="train test split at %d" % split_point, c = "b")
    plt.legend()
    plt.plot(anomaly_point, data.loc[anomaly_point, "orig"],'o')
    plt.title("%s - Anomaly Point" % (file_id))

    if not os.path.exists("./picture"):
        os.makedirs("./picture")
    plt.savefig("./picture/%s_anomaly.jpg" % (file_id))
    plt.close()


def generate_final_submission(base_path, save_path):
    """
    generate final submission file, format: No. ; Location of Anomaly
    @param base_path: path to save the ensemble results
    @param save_path: path to save the submission results
    """
    results = {}
    files = glob.glob(base_path + "/*")
    for f in files:
        file_id = int(f.split("/")[-1].split(".")[0])
        df = pd.read_csv(f)
        df = df.sort_values(["confidence"], ascending=False)
        results[file_id] = {"No.": file_id, "Location of Anomaly": int(df["idx"][0])}
    
    result_df = pd.DataFrame.from_dict(results, orient="index")
    result_df = result_df.sort_values(["No."])
    result_df.to_csv(save_path, index=False, header=True)
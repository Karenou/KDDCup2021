import glob
import pandas as pd
import numpy as np
import os

from utils import auto_period_finder
from statistic_func import *

class AnomalyDetection:

    def __init__(self, file_paths: dict, file_id: str) -> None:
        """
        @param file_paths: path to store the data
        @param file_id: 
        """
        self.file_paths = file_paths
        self.file_id = file_id
        # search for window size among this list
        self.candidates_ws = self.get_candidate_ws()

    def get_candidate_ws(self, min_ws=40, max_ws=800, rate=1.1) -> list:
        """
        @param min_ws: minimum window size
        @param max_ws: maximum window size
        @param rate: the rate at which window size increases
        return a list of candidate window sizes
        """
        # min_w * (grow_rate ** size) = max_w
        size = int(np.log(max_ws / min_ws) / np.log(rate)) + 1
        candidate_ws = [int(min_ws * (rate ** i)) for i in range(size)]
        return candidate_ws

    def load_data(self) -> tuple:
        """
        the time serie is in "orig" column
        """
        file_path = self.file_paths[self.file_id]
        # the format of these files are different
        if self.file_id in ["204","205", "206", "207", "208", "225","226", "242", "243"]:
            data = pd.read_csv(file_path, sep="\s+", header=None, names=["orig"])
            data = data.T
        else:
            data = pd.read_csv(file_path, sep=",", header=None, names=["orig"])

        point = int(file_path.split("_")[-1].split(".txt")[0])
        train = data.iloc[:point].reset_index(drop=True)
        test  = data.iloc[point:].reset_index(drop=True)

        return data, train, test, point

    def append_result(self, results: dict, name: str, ws:int, conf: float, idx: int, peak: float) -> dict:
        """
        a helper function to append result in the dict
        """
        results[f'ws={ws}, {name}'] = {"confidence": conf, "idx": idx, "peak": peak}
        return results

    def get_score_func(self):
        """
        compute the score functions of different statistical measures
        @param window size
        """
        print("loading data for file_id = %s" % self.file_id)
        data, train, test, point = self.load_data()
        data["diff"] = data["orig"].diff(1)
        data["acc"] = data["diff"].diff(1)
        results = {}

        for ws in self.candidates_ws:
        
            # print("applying fourier transformation")
            low_fft_score, high_fft_score = fourier_transform(data, ws, point)
            conf, idx, peak = compute_confidence_score(low_fft_score, ws, point)
            results = self.append_result(results, "low_fft", ws, conf, idx, peak)

            conf, idx, peak = compute_confidence_score(high_fft_score, ws, point)
            results = self.append_result(results, "high_fft", ws, conf, idx, peak)

            # print("computing score functions")
            conf, idx, peak, orig_p2p_s = peak_to_peak_value(data, "orig", ws, point)
            results = self.append_result(results, "orig_p2p", ws, conf, idx, peak) 
            
            conf, idx, peak = inverse(orig_p2p_s, ws, point)  
            results = self.append_result(results, "orig_p2p_inv", ws, conf, idx, peak) 

            conf, idx, peak, _ = peak_to_peak_value(data, "diff", ws, point)
            results = self.append_result(results, "diff_p2p", ws, conf, idx, peak)   
            
            conf, idx, peak, _ = peak_to_peak_value(data, "acc", ws, point)
            results = self.append_result(results, "acc_p2p", ws, conf, idx, peak)     
            
            conf, idx, peak = get_small_diff(data, "diff", ws, point)
            results = self.append_result(results, "diff_small", ws, conf, idx, peak)    

            conf, idx, peak, acc_std = get_std(data, "acc", ws, point)
            results = self.append_result(results, "acc_std", ws, conf, idx, peak)    

            conf, idx, peak = inverse(acc_std, ws, point)  
            results = self.append_result(results, "acc_std_inv", ws, conf, idx, peak) 

            # print("computing matrix profile")
            conf, idx, peak = orig_mp_outlier(data, ws, point)
            results = self.append_result(results, "orig_mp_outlier", ws, conf, idx, peak) 

            conf, idx, peak = orig_mp_novelty(train, test, data, ws, point)
            results = self.append_result(results, "orig_mp_novelty", ws, conf, idx, peak) 

        # save results
        result_df = pd.DataFrame.from_dict(results, orient="index")
        result_df = result_df[~result_df["confidence"].isna()].sort_values(["confidence"], ascending=False)
        
        if not os.path.exists("./results"):
            os.makedirs("./results")
        result_df.to_csv("./results/%s.csv" % self.file_id, header=True, index=True)

        # plot_anomaly
        self.plot_anomaly(data, result_df, point)

    def plot_anomaly(self, data: pd.DataFrame, result_df: pd.DataFrame, split_point: int) -> None:
        """
        plot the anomaly point against the whole time series data
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
        plt.title("%s - Anomaly Point" % (self.file_id))

        if not os.path.exists("./picture"):
            os.makedirs("./picture")
        plt.savefig("./picture/%s_anomaly.jpg" % (self.file_id))
        plt.close()
        # plt.show()


base_path = "./data-sets/KDD-Cup/data/*"
file_paths = {p.split("data/")[1].split("_")[0] : p  for p in glob.glob(base_path)}
file_ids = sorted(list(file_paths.keys()))

for file_id in file_ids:
    model = AnomalyDetection(file_paths, file_ids)
    model.get_score_func()
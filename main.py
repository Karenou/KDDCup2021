import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from fourier_transform import fourier_transform
from spectral_residual import spectral_residual
from rrcf import robust_random_cut_forrest
from matrix_profile import orig_mp_novelty, orig_mp_outlier
from statistic_func import *

from util import plot_anomaly, generate_final_submission

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
            data = pd.read_csv(file_path, sep="\s+", header=None)
            data = data.T
            data.columns = ["orig"]
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

    def get_score_func(self, plot=False):
        """
        compute the score functions of different statistical measures
        @param plot: whether to plot the anomaly picture
        """
        print("loading data for file_id = %s" % self.file_id)
        data, train, test, point = self.load_data()
        data["diff"] = data["orig"].diff(1).fillna(method="bfill")
        data["acc"] = data["diff"].diff(1).fillna(method="bfill")
        results = {}

        # grid search for hyperparameter for rrcf
        num_trees = [20, 40, 60]
        tree_sizes = [64, 128, 256]

        for ws in self.candidates_ws:
        
            # applying fourier transformation
            low_fft_score, high_fft_score = fourier_transform(data, ws)
            conf, idx, peak = compute_confidence_score(low_fft_score, ws, point)
            results = self.append_result(results, "low_fft", ws, conf, idx, peak)

            conf, idx, peak = compute_confidence_score(high_fft_score, ws, point)
            results = self.append_result(results, "high_fft", ws, conf, idx, peak)

            # apply spetral residual
            conf, idx, peak = spectral_residual(data, "orig", ws, point)
            results = self.append_result(results, "sr_orig", ws, conf, idx, peak)

            conf, idx, peak = spectral_residual(data, "diff", ws, point)
            results = self.append_result(results, "sr_diff", ws, conf, idx, peak)

            conf, idx, peak = spectral_residual(data, "acc", ws, point)
            results = self.append_result(results, "sr_acc", ws, conf, idx, peak)

            # apply rrcf
            for num_tree in num_trees:
                for tree_size in tree_sizes:
                    conf, idx, peak = robust_random_cut_forrest(data, "orig", ws, point, num_tree, tree_size)
                    results = self.append_result(results, "rrcf_orig_%d_%d" % (num_tree, tree_size), ws, conf, idx, peak)

                    conf, idx, peak = robust_random_cut_forrest(data, "diff", ws, point, num_tree, tree_size)
                    results = self.append_result(results, "rrcf_diff_%d_%d" % (num_tree, tree_size), ws, conf, idx, peak)

                    conf, idx, peak = robust_random_cut_forrest(data, "acc", ws, point, num_tree, tree_size)
                    results = self.append_result(results, "rrcf_acc_%d_%d" % (num_tree, tree_size), ws, conf, idx, peak)

            # apply statistic function
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

            # apply computing matrix profile
            conf, idx, peak = orig_mp_outlier(data, ws, point)
            results = self.append_result(results, "orig_mp_outlier", ws, conf, idx, peak) 

            conf, idx, peak = orig_mp_novelty(train, test, data, ws, point)
            results = self.append_result(results, "orig_mp_novelty", ws, conf, idx, peak) 

        # save results
        result_df = pd.DataFrame.from_dict(results, orient="index")
        result_df = result_df[~result_df["confidence"].isna()].sort_values(["confidence"], ascending=False)
        
        if not os.path.exists("./ensemble_results"):
            os.makedirs("./ensemble_results")
        result_df.to_csv("./ensemble_results/%s.csv" % self.file_id, header=True, index=True)

        if plot:
            plot_anomaly(self.file_id, data, result_df, point)

base_path = "./data-sets/KDD-Cup/data/*"
file_paths = {p.split("data/")[1].split("_")[0] : p  for p in glob.glob(base_path)}
file_ids = sorted(list(file_paths.keys()))

for file_id in file_ids:
    model = AnomalyDetection(file_paths, file_id)
    model.get_score_func(plot=False)

generate_final_submission("./ensemble_results", "submission.csv")

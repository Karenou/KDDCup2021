import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def auto_period_finder(arr, candidate_ws):
    """
    @param arr: time series
    @param candidate_ws: a list of window size candidates
    a function return the best period of the given time series among the candidates
    """
    scores = []
    
    for w in candidate_ws:
        peaks = signal.find_peaks(arr.reshape(-1), distance=w)[0] 
        valleys = signal.find_peaks(-arr.reshape(-1), distance=w)[0] 
        # calculate interval lengths pd and vd from peaks and valleys
        pd = np.mean(np.diff(peaks))
        vd = np.mean(np.diff(valleys))
        s_d = min(pd, vd) / np.sqrt(w)
        scores.append([w, s_d])
    
    # find argmin scores
    idx = np.argmin(np.array(scores)[:,1])
    return scores[idx][0]


def plot_anomaly(data, df, name, file_id):
    plt.figure(figsize=(20,4))
    plt.plot(np.arange(len(data)), data["orig"])

    a = np.array(df[df["confidence"] == df["confidence"].max()]["idx"])
    a.sort()
    plt.plot(a, data.loc[a, "orig"],'o')
    plt.title("%s - %s" % (file_id, name))

    plt.savefig("./picture/%s_%s.jpg" % (file_id, name))
    plt.show()

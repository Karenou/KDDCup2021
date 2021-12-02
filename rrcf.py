import pandas as pd
from rrcf_src import RCTree, shingle
from util import smoothing, compute_confidence_score

def robust_random_cut_forrest(arr: pd.DataFrame, name: str, ws: int, split_point: int,
                            num_trees: int, tree_size: int) -> tuple:
    
    #use rrcf method to find the anomaly
    print("for rrcf computing....")
    shingle_size = ws

    data_list=list(arr[name].values)
    forest = []
    for _ in range(num_trees):
        tree = RCTree()
        forest.append(tree)
        
    # Use the "shingle" generator to create rolling window
    points = shingle(data_list, size=shingle_size)
    
    # Create a dict to store anomaly score of each point
    avg_codisp = {}
    
    # For each shingle...
    for index, point in enumerate(points):
        # For each tree in the forest...
        for tree in forest:
            # If tree is above permitted size, drop the oldest point (FIFO)
            if len(tree.leaves) > tree_size:
                tree.forget_point(index - tree_size)
            # Insert the new point into the tree
            tree.insert_point(point, index=index)
            # Compute codisp on the new point and take the average among all trees
            if not index in avg_codisp:
                avg_codisp[index] = 0
            avg_codisp[index] += tree.codisp(index) / num_trees
    
    score = avg_codisp.values()

    #avg_codisp = spec.generate_anomaly_score(arr[name].values)
    score = smoothing(score, ws)
    conf, idx, peak = compute_confidence_score(score, ws, split_point)
    return conf, idx, peak
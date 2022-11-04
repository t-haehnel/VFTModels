import numpy as np
import pandas as pd


def get_cluster_statistics(words: pd.DataFrame) -> dict:
    """
    Calculates some general clustering features (see return-description).
    :param words: pd.DataFrame containing a column "cluster" with np.NAN for words without a cluster and an integer for
    words within a cluster (same value for all words belonging to the same cluster)
    :return: dict with keys mean (mean cluster size), median (median cluster size), max (max cluster size), count
    (number of clusters) and count_per_size (dictionary, key: occuring cluster sizes, value: number of clusters with
    thisl size
    """

    # count cluster occurences
    cluster_size_dict = {}
    cluster_size_list = []
    cluster_curr_id = np.NAN
    cluster_curr_count = 0

    words = words.copy().reset_index(drop=True)

    for i in range(0, words.shape[0] + 1):
        # store clusters at end of a cluster (-> at last position or if cluster id changes)
        if (i == words.shape[0]) or (cluster_curr_id != words.loc[i]["cluster"]):
            # store the cluster if size is > 1
            if cluster_curr_count > 1:
                if cluster_curr_count not in cluster_size_dict.keys():
                    cluster_size_dict[cluster_curr_count] = 0
                cluster_size_dict[cluster_curr_count] += 1
                cluster_size_list.append(cluster_curr_count)
            # reset cluster counter
            cluster_curr_count = 0

        # increase cluster size count if 1) there is a cluster (!= np.NAN) the first condition only prevents an error
        # with the index when calculating the latter conditions
        if (i < words.shape[0]) and not np.isnan(words.loc[i]["cluster"]):
            cluster_curr_count += 1

        if i < words.shape[0]:
            cluster_curr_id = words.loc[i]["cluster"]

    # next calculate statistical parameters, then return them
    if len(cluster_size_list) > 0:
        max_size = max(cluster_size_list)
        mean_size = np.mean(cluster_size_list)
        median_size = np.median(cluster_size_list)
    else:
        max_size = 0
        mean_size = 0
        median_size= 0
    cluster_count = len(cluster_size_list)

    return {"mean": mean_size,
            "median": median_size,
            "max": max_size,
            "number_of_clusters": cluster_count,
            "count_per_size": cluster_size_dict}

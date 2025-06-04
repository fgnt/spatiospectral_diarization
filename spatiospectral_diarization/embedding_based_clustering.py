
from sklearn.cluster import HDBSCAN
import numpy as np

def embeddings_hdbscan_clustering(embeddings, seg_boundaries):
    """
    Performs speaker clustering on segment embeddings using HDBSCAN.

    This function takes a list of embeddings and their corresponding segment boundaries,
    filters out short segments (1 second), and applies HDBSCAN clustering to the remaining embeddings.
    It returns the cluster labels for the filtered (long) segments and the activities of the clustered embeddings.

    Args:
        embeddings (list or np.ndarray): List of embedding vectors for each segment.
        seg_boundaries (list of tuple): List of (onset, offset) tuples for each segment.

    Returns:
        np.ndarray: Cluster labels assigned by HDBSCAN for the filtered embeddings.
        np.ndarray: The activities of the filtered embeddings used in HDBSCAN.
    """
    embeddings_red = []
    activities_red = []
    embeddings_short = []
    activities_short = []
    for i, e in enumerate(embeddings):
        onset, offset = seg_boundaries[i]
        if offset - onset > 16000:
            embeddings_red.append(e)
            activities_red.append((onset, offset))
        else:
            embeddings_short.append(e)
            activities_short.append((onset, offset))
    labels = HDBSCAN(
        min_cluster_size=3, min_samples=3, cluster_selection_epsilon=0.,
        max_cluster_size=None, metric='cosine'
    ).fit_predict(embeddings_red)
    return labels, activities_red, embeddings_red
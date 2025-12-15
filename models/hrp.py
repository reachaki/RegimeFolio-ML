# models/hrp.py
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list


def correl_dist(corr):
    """
    Distance matrix for clustering from correlation matrix.
    d_ij = sqrt(0.5 * (1 - rho_ij))
    """
    return np.sqrt(0.5 * (1 - corr))


def get_quasi_diag(link):
    return leaves_list(link)


def get_cluster_var(cov, cluster_items):
    cov_slice = cov.loc[cluster_items, cluster_items]
    w = np.ones(len(cov_slice)) / len(cov_slice)
    return float(np.dot(w, np.dot(cov_slice.values, w)))


def hrp_allocation(cov):
    """
    Hierarchical Risk Parity weights from covariance matrix (DataFrame).
    """
    corr = cov.corr()
    dist = correl_dist(corr)

    # Condensed distance for linkage
    # SciPy expects a condensed distance matrix; use squareform-style flatten
    from scipy.spatial.distance import squareform

    dist_condensed = squareform(dist.values, checks=False)
    link = linkage(dist_condensed, "ward")

    sort_ix = get_quasi_diag(link)
    sort_labels = corr.index[sort_ix]
    cov_sorted = cov.loc[sort_labels, sort_labels]

    # Recursive bisection
    weights = pd.Series(1.0, index=sort_labels)
    clusters = [sort_labels.tolist()]

    while len(clusters) > 0:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            c1 = cluster[:split]
            c2 = cluster[split:]
            new_clusters += [c1, c2]

            var1 = get_cluster_var(cov_sorted, c1)
            var2 = get_cluster_var(cov_sorted, c2)
            alpha = 1 - var1 / (var1 + var2)
            weights[c1] *= alpha
            weights[c2] *= 1 - alpha
        clusters = new_clusters

    # Normalise
    weights /= weights.sum()
    # Reindex back to original order
    return weights.reindex(cov.index)

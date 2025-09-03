import torch
from torch import Tensor

from typing import Tuple
from utils._distance import euclidean_distance, manhattan_distance, chebyshev_distance, minkowski_distance


def knn_graph(X: Tensor, k: int, metric: str = "euclidean", p: int = 3) -> Tuple[Tensor, Tensor]:
    """
    Compute k-nearest neighbors graph.
    
    :param X: Input data tensor with shape [n_samples, n_features]
    :param k: Number of nearest neighbors
    :param metric: Distance metric ('euclidean', 'manhattan', 'chebyshev', 'minkowski')
    :param p: Power parameter for Minkowski distance
    :return: Tuple of (distances, indices) tensors
    """
    n_samples = X.shape[0]

    if metric == "euclidean":
        dist_func = euclidean_distance
    elif metric == "manhattan":
        dist_func = manhattan_distance
    elif metric == "chebyshev":
        dist_func = chebyshev_distance
    elif metric == "minkowski":
        dist_func = minkowski_distance
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    distances = torch.zeros((n_samples, n_samples), device=X.device, dtype=X.dtype)
    for i in range(n_samples):
        if metric == "minkowski":
            distances[i] = dist_func(X, X[i], p=p)
        else:
            distances[i] = dist_func(X, X[i])

    knn_distances = torch.zeros((n_samples, k), device=X.device, dtype=X.dtype)
    knn_indices = torch.zeros((n_samples, k), device=X.device, dtype=torch.long)
    
    for i in range(n_samples):
        dists = distances[i]
        dists[i] = float('inf')
        knn_distances[i], knn_indices[i] = torch.topk(dists, k, largest=False)
    
    return knn_distances, knn_indices

"""
用于存放和记录多种不同距离的计算函数

Created on 2025/08/20 16:09:21
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan

动态时间扭曲 DTW

DTW + KNN
"""

import torch
from sys import maxsize

from typing import Optional, Callable


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean distance between batches of time series.

    Advantages:
    (1) Extremely fast computation with O(n) complexity.
    (2) Easy to understand and implement.

    Disadvantages:
    (1) Highly sensitive to noise and outliers.
    (2) Requires sequences to have equal length.
    (3) Cannot handle time axis "stretching" or "compression" (i.e., phase shift).
        For example, if one sequence is a translation of another, the Euclidean distance
        will be large even if their shapes are identical.

    :param x: Input time series tensor with shape [n_samples, n_length].
    :param y: Input time series tensor with shape [n_samples, n_length] or [1, n_length] (for broadcasting).
    :return: Euclidean distance between x and y as a tensor with shape [n_samples].
    """
    return torch.norm(x - y, p=2, dim=1)


def manhattan_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Manhattan distance (City Block distance) between batches of time series.

    Advantages:
    (1) Fast computation with O(n) complexity.
    (2) Slightly more robust to outliers than Euclidean distance (uses absolute difference instead of squared).

    Disadvantages:
    (1) Requires sequences to have equal length.
    (2) Cannot handle time axis "stretching" or "compression" (i.e., phase shift).

    :param x: Input time series tensor with shape [n_samples, n_length].
    :param y: Input time series tensor with shape [n_samples, n_length] or [1, n_length] (for broadcasting).
    :return: Manhattan distance between x and y as a tensor with shape [n_samples].
    """
    return torch.sum(torch.abs(x - y), dim=1)


def chebyshev_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the Chebyshev distance (maximum norm) between batches of time series.

    Advantages:
    (1) Fast computation with O(n) complexity.
    (2) Focuses on the worst-case pointwise difference.

    Disadvantages:
    (1) Ignores information from other points, determined solely by one pair of points.
    (2) Requires sequences to have equal length.
    (3) Cannot handle time axis "stretching" or "compression" (i.e., phase shift).

    :param x: Input time series tensor with shape [n_samples, n_length].
    :param y: Input time series tensor with shape [n_samples, n_length] or [1, n_length] (for broadcasting).
    :return: Chebyshev distance between x and y as a tensor with shape [n_samples].
    """
    return torch.max(torch.abs(x - y), dim=1)[0]


def minkowski_distance(x: torch.Tensor, y: torch.Tensor, p: float = 3) -> torch.Tensor:
    """
    Compute the Minkowski distance between batches of time series.

    Advantages:
    (1) Provides a family of distance metrics parameterized by p.
    (2) Generalizes both Manhattan (p=1) and Euclidean (p=2) distances.

    Disadvantages:
    (1) Requires sequences to have equal length.
    (2) Cannot handle time axis "stretching" or "compression" (i.e., phase shift).
    (3) Choice of p parameter adds complexity to model selection.

    :param x: Input time series tensor with shape [n_samples, n_length].
    :param y: Input time series tensor with shape [n_samples, n_length] or [1, n_length] (for broadcasting).
    :param p: Distance parameter (1=Manhattan, 2=Euclidean, ∞=Chebyshev).
    :return: Minkowski distance between x and y as a tensor with shape [n_samples].
    """
    if p == float("inf"):
        return chebyshev_distance(x, y)
    return torch.sum(torch.abs(x - y) ** p, dim=1) ** (1 / p)


def dtw_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    d: Callable = lambda x, y: torch.abs(x - y),
    max_warping_window: Optional[int] = 10000,
) -> torch.Tensor:
    """
    Returns the DTW similarity distance between two 2-D timeseries numpy arrays.

    :param x: array of shape [n_samples, n_timepoints]
    :param y: array of shape [n_samples, n_timepoints]
                 Two arrays containing n_samples of timeseries data
                 whose DTW distance between each sample of A and B
                 will be compared
    :param d: DistanceMetric object (default = abs(x-y)) the distance measure used for A_i - B_j in the
              DTW dynamic programming function
    :param max_warping_window: int, optional (default = infinity)
                               Maximum warping window allowed by the DTW dynamic programming function
    :return: DTW distance between A and B
    """
    # Create cost matrix via broadcasting with large int
    M, N = len(x), len(y)
    cost = maxsize * torch.ones((M, N))

    # Initialize the first row and column
    print(cost)
    cost[0, 0] = d(x[0], y[0])

    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(x[i], y[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(x[0], y[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window), min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = torch.min(choices) + d(x[i], y[j])

    # Return DTW distance given window
    return cost[-1, -1]


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    n_samples = 3
    n_length = 5
    x = torch.randn(n_samples, n_length)
    y = torch.randn(n_samples, n_length)

    print("Input x:")
    print(x)
    print("\nInput y:")
    print(y)

    # Compute various distances
    print("\nEuclidean distance:")
    print(euclidean_distance(x, y))

    print("\nManhattan distance:")
    print(manhattan_distance(x, y))

    print("\nChebyshev distance:")
    print(chebyshev_distance(x, y))

    print("\nMinkowski distance (p=1, equivalent to Manhattan):")
    print(minkowski_distance(x, y, p=1))

    print("\nMinkowski distance (p=2, equivalent to Euclidean):")
    print(minkowski_distance(x, y, p=2))

    print("\nMinkowski distance (p=3):")
    print(minkowski_distance(x, y, p=3))

    print("\nMinkowski distance (p=∞, equivalent to Chebyshev):")
    print(minkowski_distance(x, y, p=float("inf")))

# -*- coding: utf-8 -*-
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

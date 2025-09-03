import torch
from torch import Tensor


def linear_kernel(X: Tensor, Y: Tensor) -> Tensor:
    """Linear kernel"""
    return X @ Y.T


def polynomial_kernel(X: Tensor, Y: Tensor, degree: int = 3, gamma: float = None, coef0: float = 1) -> Tensor:
    """Polynomial kernel"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = X @ Y.T
    K *= gamma
    K += coef0
    return K.pow(degree)


def rbf_kernel(X: Tensor, Y: Tensor, gamma: float = None) -> Tensor:
    """RBF kernel"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    X_norm = torch.sum(X**2, dim=1, keepdim=True)
    Y_norm = torch.sum(Y**2, dim=1, keepdim=True)
    K = X_norm - 2 * X @ Y.T + Y_norm.T
    K = torch.exp(-gamma * K)
    return K


def sigmoid_kernel(X: Tensor, Y: Tensor, gamma: float = None, coef0: float = 1) -> Tensor:
    """Sigmoid kernel"""
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = X @ Y.T
    K *= gamma
    K += coef0
    return torch.tanh(K)


def cosine_similarity_kernel(X: Tensor, Y: Tensor) -> Tensor:
    """Cosine similarity kernel"""
    X_norm = torch.norm(X, dim=1, keepdim=True)
    Y_norm = torch.norm(Y, dim=1, keepdim=True)
    K = X @ Y.T
    K = K / (X_norm @ Y_norm.T)
    return K

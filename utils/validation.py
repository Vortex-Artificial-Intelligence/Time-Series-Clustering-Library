import torch
from torch import Tensor

from numpy import ndarray
from typing import Union


def check_array(
    X: Union[ndarray, Tensor],
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Input validation for standard estimators"""
    if not isinstance(X, Tensor):
        X = torch.tensor(X, dtype=dtype, device=device)
    else:
        X = X.to(dtype=dtype, device=device)

    if X.ndim != 2:
        raise ValueError("Expected 2D array, got {}D array instead".format(X.ndim))

    return X


def check_random_state(seed: Union[int, None]) -> torch.Generator:
    """Turn seed into a torch.Generator instance"""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator

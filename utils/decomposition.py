import torch
from torch import Tensor

from typing import Tuple


def svd_flip(u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    """Sign correction to ensure deterministic output from SVD"""
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), dim=0)
    signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

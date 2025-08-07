import numpy as np
import torch

from numpy import ndarray
from torch import Tensor

from typing import Optional

from models.base import BaseClustering


class KMeans(BaseClustering):
    """KMeans clustering model"""

    def __init__(self, n_clusters: int,
        cpu: Optional[bool] = False,
        device: Optional[int] = 0,
        dtype: Optional[torch.dtype] = torch.float64,
        random_state: Optional[int] = 42,) -> None:
        super().__init__(n_clusters=n_clusters, cpu=cpu, device=device, dtype=dtype, random_state=random_state)
        pass

    def fit(self, X: ndarray | Tensor, *args) -> None:
        pass

    def predict(self, X: ndarray | Tensor) -> ndarray | Tensor:
        pass


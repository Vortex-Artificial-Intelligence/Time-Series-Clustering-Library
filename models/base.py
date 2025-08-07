import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from utils import set_cuda_device, set_torch_dtype


class BaseModel(ABC):
    """The abstract base class for all models"""

    def __init__(
        self,
        device: Optional[int] = 0,
        dtype: Optional[torch.dtype] = torch.float64,
        random_state: Optional[int] = 42,
    ) -> None:
        self.cuda_index = device
        self.device = self.set_cuda_device()

        self.dtype = dtype
        self.set_cuda_device()

        self.random_state = random_state

        # Record the data entered by the user and convert it into the same format later
        self.inputs_type = None

    def __str__(self) -> str:
        """Get the string representation of the model"""
        return "BaseModel"

    def set_cuda_device(self) -> torch.cuda.device:
        """Set the CUDA device to accelerate computing"""
        return set_cuda_device(cpu=self.cpu, cuda_index=self.cuda_index)

    def set_torch_dtype(self) -> None:
        """Set the PyTorch dtype for global"""
        set_torch_dtype(self.dtype)

    def ndarray2tensor(self, x: ndarray) -> Tensor:
        """
        Convert a numpy array to torch.tensor.
        
        :param x: the input numpy array.
        :return: the converted torch tensor.
        """ ""
        return torch.from_numpy(x).to(self.device)

    @staticmethod
    def tensor2ndarray(x: Tensor) -> ndarray:
        """
        Convert a torch tensor to numpy array.

        :param x: the input torch tensor.
        :return: the converted numpy array.
        """
        return x.cpu().detach().numpy()


class BaseClustering(BaseModel):
    """The Base Class Inherited by Clustering Models"""

    def __init__(
        self,
        n_clusters: int,
        cpu: Optional[bool] = False,
        device: Optional[int] = 0,
        dtype: Optional[torch.dtype] = torch.float64,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            self, cpu=cpu, device=device, dtype=dtype, random_state=random_state
        )

        self.n_clusters = n_clusters

    def __str__(self) -> str:
        """Get the name of this cluster"""
        return "BaseClustering"

    def set_cuda_device(self) -> torch.cuda.device:
        """Set the CUDA device to accelerate computing"""
        return set_cuda_device(cpu=self.cpu, cuda_index=self.cuda_index)

    def set_torch_dtype(self) -> None:
        """Set the PyTorch dtype for global"""
        set_torch_dtype(self.dtype)

    @abstractmethod
    def fit(self, X: ndarray | Tensor, *args) -> None:
        """
        Fit the clustering model to the given data.

        :param X: the input data to be clustered, ndarray or Tensor.
        :param args: other arguments passed to the fit method.
        :return: None.
        """
        pass

    @abstractmethod
    def predict(self, X: ndarray | Tensor, *args) -> ndarray | Tensor:
        """
        Predict the cluster labels for the given data.

        :param X: the input data to be clustered, ndarray or Tensor.
        :param args: other arguments passed to the fit method.
        :return: the cluster labels for the given data of ndarray or Tensor.
        """
        pass


class BaseDimensionalityReduction(BaseModel):
    """The Base Class Inherited by Dimensionality Reduction Models"""

    def __init__(
        self,
        n_components: int,
        cpu: Optional[bool] = False,
        device: Optional[int] = 0,
        dtype: Optional[torch.dtype] = torch.float64,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            self, cpu=cpu, device=device, dtype=dtype, random_state=random_state
        )
        self.n_components = n_components

    def __str__(self) -> str:
        """Get the name of this cluster"""
        return "BaseDimensionalityReduction"

    @abstractmethod
    def fit_transform(self, X: ndarray | Tensor, *args) -> ndarray | Tensor:
        """
        Execute dimensionality reduction algorithm.

        :param X: the input numpy array or torch tensor to be reduced.
        :param args:
        :return:
        """
        pass

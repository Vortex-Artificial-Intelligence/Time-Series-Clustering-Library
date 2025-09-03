import numpy as np
from numpy import ndarray

import torch
from torch import Tensor

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from utils.tools import set_cuda_device, set_torch_dtype, set_random_state


class BaseModel(ABC):
    """The abstract base class for all models"""

    def __init__(
        self,
        cpu: Optional[bool] = False,
        device: Optional[int] = 0,
        dtype: Optional[torch.dtype] = torch.float64,
        random_state: Optional[int] = 42,
    ) -> None:
        self.cpu = cpu
        self.cuda_index = device
        self.device = self.set_cuda_device()
        self.dtype = dtype
        self.set_torch_dtype()
        self.random_state = random_state
        self.set_random_state()

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

    def set_random_state(self) -> None:
        """Set the random state for reproducibility"""
        set_random_state(self.random_state)

    def ndarray2tensor(self, x: ndarray) -> Tensor:
        """
        Convert a numpy array to torch.tensor.
        
        :param x: the input numpy array.
        :return: the converted torch tensor.
        """
        return torch.from_numpy(x).to(self.device).to(self.dtype)

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
        distance: Optional[str] = "euclidean",
        cpu: Optional[bool] = False,
        device: Optional[int] = 0,
        dtype: Optional[torch.dtype] = torch.float64,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(cpu=cpu, device=device, dtype=dtype, random_state=random_state)

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

    def set_random_state(self) -> None:
        """Set the random state for reproducibility"""
        set_random_state(self.random_state)

    @abstractmethod
    def fit(self, X: ndarray | Tensor) -> None:
        """
        Fit the clustering model to the given data.

        :param X: the input data to be clustered, ndarray or Tensor.
        :param args: other arguments passed to the fit method.
        :return: None.
        """
        pass

    @abstractmethod
    def predict(self, X: ndarray | Tensor) -> ndarray | Tensor:
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
        super().__init__(cpu=cpu, device=device, dtype=dtype, random_state=random_state)
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def __str__(self) -> str:
        """Get the name of this cluster"""
        return "BaseDimensionalityReduction"

    def check_input(self, X: Union[ndarray, Tensor]) -> Tensor:
        """
        Check and convert input to tensor with proper dtype and device.
        
        :param X: input data
        :return: converted tensor
        """
        if isinstance(X, ndarray):
            X = self.ndarray2tensor(X)
        elif isinstance(X, Tensor):
            X = X.to(device=self.device, dtype=self.dtype)
        else:
            raise ValueError("Input must be numpy array or torch tensor")
        
        if X.ndim != 2:
            raise ValueError("Expected 2D array, got {}D array instead".format(X.ndim))
            
        return X

    @abstractmethod
    def fit(self, X: Union[ndarray, Tensor]) -> "BaseDimensionalityReduction":
        """Fit the model with X"""
        pass

    @abstractmethod
    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Apply dimensionality reduction to X"""
        pass

    def fit_transform(self, X: Union[ndarray, Tensor], *args) -> Union[ndarray, Tensor]:
        """Fit the model with X and apply the dimensionality reduction on X"""
        self.fit(X)
        return self.transform(X)

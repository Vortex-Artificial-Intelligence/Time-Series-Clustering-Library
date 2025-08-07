import random
import numpy as np
import torch

from typing import Optional


def set_random_state(seed: Optional[int] = 42) -> None:
    """
    Set random seed for reproducibility in deep learning model training.

    :param seed: the random seed for `random`, `numpy`, and `torch`.`
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cuda_device(
    cpu: Optional[bool] = False, cuda_index: Optional[int] = 0
) -> torch.cuda.device:
    """
    Set the CUDA device to accelerate computing.

    :param cpu: whether to use CPU device only, defaults False for using CUDA.
    :param cuda_index:  the index of CUDA device to use, defaults 0.
    :return: the CUDA device of torch.
    """
    return torch.device(
        f"cuda:{cuda_index}" if torch.cuda.is_available() and cpu is False else "cpu"
    )


def set_torch_dtype(dtype: torch.dtype = torch.float32) -> None:
    """
    Set the PyTorch dtype for running global.

    :param dtype: the dtype of torch.Tensor, defaults to torch.float32.
    :return: None.
    """
    torch.set_default_dtype(dtype)

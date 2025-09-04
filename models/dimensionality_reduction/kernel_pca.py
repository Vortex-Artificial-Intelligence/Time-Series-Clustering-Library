import torch
from torch import Tensor

import numpy as np
from numpy import ndarray
from typing import Union, Optional, Callable

from models.base import BaseDimensionalityReduction
from utils.kernels import (
    linear_kernel,
    rbf_kernel,
    polynomial_kernel,
    sigmoid_kernel,
    cosine_similarity_kernel,
)


class KernelPCA(BaseDimensionalityReduction):
    """Kernel Principal Component Analysis (KernelPCA)"""

    def __init__(
        self,
        n_components: int,
        kernel: Union[str, Callable] = "linear",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1,
        kernel_params: Optional[dict] = None,
        alpha: float = 1e-6,
        cpu: Optional[bool] = False,
        device: Optional[int] = 0,
        dtype: Optional[torch.dtype] = torch.float64,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            n_components=n_components,
            cpu=cpu,
            device=device,
            dtype=dtype,
            random_state=random_state,
        )
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params or {}
        self.alpha = alpha
        self.X_fit_ = None
        self.lambdas_ = None
        self.alphas_ = None

    def __str__(self) -> str:
        return "KernelPCA"

    def _get_kernel(self, kernel):
        """Get kernel function"""
        if callable(kernel):
            return kernel
        elif kernel == "linear":
            return linear_kernel
        elif kernel == "poly" or kernel == "polynomial":
            return lambda X, Y: polynomial_kernel(
                X, Y, degree=self.degree, gamma=self.gamma, coef0=self.coef0
            )
        elif kernel == "rbf":
            return lambda X, Y: rbf_kernel(X, Y, gamma=self.gamma)
        elif kernel == "sigmoid":
            return lambda X, Y: sigmoid_kernel(X, Y, gamma=self.gamma, coef0=self.coef0)
        elif kernel == "cosine":
            return cosine_similarity_kernel
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    def fit(self, X: Union[ndarray, Tensor]) -> "KernelPCA":
        """Fit the model with X"""
        X = self.check_input(X)
        self.X_fit_ = X.clone()

        # Get kernel function
        kernel_fn = self._get_kernel(self.kernel)

        # Compute kernel matrix
        K = kernel_fn(X, X)

        # Center kernel matrix
        n_samples = X.shape[0]
        one_n = (
            torch.ones((n_samples, n_samples), dtype=self.dtype, device=self.device)
            / n_samples
        )
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        # Add small regularization for numerical stability
        K_centered += (
            torch.eye(n_samples, dtype=self.dtype, device=self.device) * self.alpha
        )

        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(K_centered)

        # Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Remove eigenvalues with negative values (due to numerical errors)
        non_negative_idx = eigenvalues > 0
        eigenvalues = eigenvalues[non_negative_idx]
        eigenvectors = eigenvectors[:, non_negative_idx]

        # Normalize eigenvectors
        self.alphas_ = eigenvectors / torch.sqrt(eigenvalues).unsqueeze(0)

        # Store top eigenvalues and eigenvectors
        self.lambdas_ = eigenvalues[: self.n_components]
        self.alphas_ = self.alphas_[:, : self.n_components]

        return self

    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Apply dimensionality reduction to X"""
        if self.alphas_ is None:
            raise RuntimeError("KernelPCA must be fitted before transforming data")

        X = self.check_input(X)

        # Get kernel function
        kernel_fn = self._get_kernel(self.kernel)

        # Compute kernel matrix between X and training data
        K = kernel_fn(X, self.X_fit_)

        # Center kernel matrix
        n_samples_train = self.X_fit_.shape[0]
        n_samples_test = X.shape[0]
        one_n_train = (
            torch.ones(
                (n_samples_train, n_samples_train), dtype=self.dtype, device=self.device
            )
            / n_samples_train
        )
        one_n_test = (
            torch.ones(
                (n_samples_test, n_samples_train), dtype=self.dtype, device=self.device
            )
            / n_samples_train
        )

        K_centered = (
            K
            - one_n_test @ kernel_fn(self.X_fit_, self.X_fit_)
            - K @ one_n_train
            + one_n_test @ kernel_fn(self.X_fit_, self.X_fit_) @ one_n_train
        )

        # Project data
        X_transformed = K_centered @ self.alphas_

        return X_transformed


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples = 1000
    t = torch.linspace(0, 4 * np.pi, n_samples)
    X = torch.stack([torch.sin(t), torch.cos(t), t], dim=1)
    X += 0.1 * torch.randn(n_samples, 3)

    # Test KernelPCA with RBF kernel
    kpca = KernelPCA(n_components=1, kernel="rbf", gamma=0.1)
    X_transformed = kpca.fit_transform(X)

    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)

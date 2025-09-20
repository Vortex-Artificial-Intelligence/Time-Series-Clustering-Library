import torch
from torch import Tensor

from numpy import ndarray
from typing import Union, Optional

from models.base import BaseDimensionalityReduction
from utils.decomposition import svd_flip


class PCA(BaseDimensionalityReduction):
    """
    Principal Component Analysis (PCA)

    A linear dimensionality reduction technique that uses Singular Value Decomposition (SVD)
    to project data to a lower-dimensional space while preserving as much variance as possible.

    Parameters:
    -----------
    n_components : int
        Number of principal components to keep
    whiten : bool, default=False
        Whether to transform data to unit variance after projection
    cpu : bool, optional, default=False
        If True, forces computation on CPU
    device : int, optional, default=0
        CUDA device index for GPU computation
    dtype : torch.dtype, optional, default=torch.float64
        Data type for tensor computations
    random_state : int, optional, default=42
        Random seed for reproducibility

    Attributes:
    -----------
    components_ : Tensor of shape (n_components, n_features)
        Principal components (eigenvectors)
    explained_variance_ : Tensor of shape (n_components,)
        Variance explained by each selected component
    explained_variance_ratio_ : Tensor of shape (n_components,)
        Percentage of variance explained by each component
    singular_values_ : Tensor of shape (n_components,)
        Singular values corresponding to each component
    mean_ : Tensor of shape (n_features,)
        Mean of training data
    """

    def __init__(
        self,
        n_components: int,
        whiten: bool = False,
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
        self.whiten = whiten
        self.singular_values_ = None

    def __str__(self) -> str:
        return "PCA"

    def fit(self, X: Union[ndarray, Tensor]) -> "PCA":
        """Fit the model with X"""
        X = self.check_input(X)

        # Center data
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_

        # Perform SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        # Flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        # Components and explained variance
        self.components_ = Vt[: self.n_components]
        self.explained_variance_ = (S**2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / torch.sum(
            self.explained_variance_
        )
        self.singular_values_ = S[: self.n_components]

        return self

    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Apply dimensionality reduction to X"""
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted before transforming data")

        X = self.check_input(X)
        X_centered = X - self.mean_

        # Project data
        X_transformed = X_centered @ self.components_.T

        # Whitening if required
        if self.whiten:
            X_transformed /= torch.sqrt(self.explained_variance_[: self.n_components])

        return X_transformed

    def inverse_transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Transform data back to its original space"""
        if self.components_ is None:
            raise RuntimeError("PCA must be fitted before inverse transforming data")

        X = self.check_input(X)

        if self.whiten:
            X = X * torch.sqrt(self.explained_variance_[: self.n_components])

        X_original = X @ self.components_ + self.mean_
        return X_original


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.randn(1000, 100)

    # Test PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Test inverse transform
    X = X.to(pca.device)
    X_reconstructed = pca.inverse_transform(X_transformed)
    print("Reconstruction error:", torch.mean((X - X_reconstructed) ** 2).item())

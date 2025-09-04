import torch
from torch import Tensor

from numpy import ndarray
from typing import Union, Optional

from utils.decomposition import svd_flip
from models.base import BaseDimensionalityReduction


class IncrementalPCA(BaseDimensionalityReduction):
    """Incremental Principal Component Analysis (IncrementalPCA)"""

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
        self.var_ = None
        self.total_samples_seen_ = 0
        self.iterated = False
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def __str__(self) -> str:
        return "IncrementalPCA"

    def fit(self, X: Union[ndarray, Tensor]) -> "IncrementalPCA":
        """Fit the model with X"""
        X = self.check_input(X)
        self.partial_fit(X)

        return self

    def partial_fit(self, X: Union[ndarray, Tensor]) -> "IncrementalPCA":
        """Incremental fit with X"""
        X = self.check_input(X)
        n_samples, n_features = X.shape

        if self.components_ is None:
            self.components_ = torch.zeros(
                (self.n_components, n_features), dtype=self.dtype, device=self.device
            )

        if self.mean_ is None:
            self.mean_ = torch.zeros(n_features, dtype=self.dtype, device=self.device)

        if self.var_ is None:
            self.var_ = torch.zeros(n_features, dtype=self.dtype, device=self.device)

        if self.singular_values_ is None:
            self.singular_values_ = torch.zeros(
                self.n_components, dtype=self.dtype, device=self.device
            )

        # Update mean and variance
        col_mean = torch.mean(X, dim=0)
        col_var = torch.var(X, dim=0, unbiased=False)

        # First pass
        if not self.iterated:
            self.mean_ = col_mean
            self.var_ = col_var
            self.total_samples_seen_ = n_samples
            self.iterated = True
        else:
            # Update mean and variance using Welford's algorithm
            total_samples = self.total_samples_seen_ + n_samples
            mean_old = self.mean_.clone()
            self.mean_ = (
                self.total_samples_seen_ * self.mean_ + n_samples * col_mean
            ) / total_samples

            # Update variance using the correct formula
            self.var_ = (
                self.total_samples_seen_ * self.var_
                + n_samples * col_var
                + self.total_samples_seen_ * (mean_old - self.mean_) ** 2
                + n_samples * (col_mean - self.mean_) ** 2
            ) / total_samples

            self.total_samples_seen_ = total_samples

        X_centered = X - self.mean_

        # Update components using incremental SVD
        if self.total_samples_seen_ >= self.n_components:
            if torch.any(self.singular_values_ != 0):
                X_proj = X_centered @ self.components_.T

                diag_s = torch.diag(self.singular_values_)
                H = torch.cat([diag_s, X_proj], dim=0)

                U, S, Vt = torch.linalg.svd(H, full_matrices=False)

                U, Vt = svd_flip(U, Vt)

                self.components_ = Vt[: self.n_components] @ self.components_
                self.singular_values_ = S[: self.n_components]
            else:
                U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

                U, Vt = svd_flip(U, Vt)

                self.components_ = Vt[: self.n_components]
                self.singular_values_ = S[: self.n_components]

            # Compute explained variance
            self.explained_variance_ = (self.singular_values_**2) / (
                self.total_samples_seen_ - 1
            )
            total_var = torch.sum(self.var_)
            if total_var > 0:
                self.explained_variance_ratio_ = self.explained_variance_ / total_var
            else:
                self.explained_variance_ratio_ = torch.zeros_like(
                    self.explained_variance_
                )

        return self

    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Apply dimensionality reduction to X"""
        if self.components_ is None:
            raise RuntimeError("IncrementalPCA must be fitted before transforming data")

        X = self.check_input(X)
        X_centered = X - self.mean_

        # Project data
        X_transformed = X_centered @ self.components_.T

        # Whitening if required
        if self.whiten and self.explained_variance_ is not None:
            # Avoid division by zero
            nonzero_variance = self.explained_variance_ > 1e-12
            if torch.any(nonzero_variance):
                X_transformed[:, nonzero_variance] /= torch.sqrt(
                    self.explained_variance_[nonzero_variance].unsqueeze(0)
                )

        return X_transformed


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.randn(1000, 100)

    ipca = IncrementalPCA(10)
    X_transformed = ipca.fit_transform(X)

    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)
    print("Explained variance ratio:", ipca.explained_variance_ratio_)

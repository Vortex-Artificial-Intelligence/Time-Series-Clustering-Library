import torch
from torch import Tensor

from numpy import ndarray
from typing import Union, Optional, Tuple

from models.base import BaseDimensionalityReduction
from utils.distance import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
)


class MDS(BaseDimensionalityReduction):
    """
    Multidimensional Scaling (MDS)

    A technique for dimensionality reduction that attempts to preserve pairwise distances
    between data points in the low-dimensional representation.

    Parameters:
    -----------
    n_components : int
        Number of dimensions in the embedded space
    metric : bool, default=True
        If True, perform metric MDS; otherwise, perform non-metric MDS
    dissimilarity : str, default='euclidean'
        Dissimilarity measure to use ('euclidean', 'manhattan', 'chebyshev', 'minkowski', 'precomputed')
    p : int, default=3
        Distance parameter for minkowski, (1=Manhattan, 2=Euclidean, ∞=Chebyshev)
    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different initializations
    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm
    eps : float, default=1e-3
        Relative tolerance with respect to stress to declare convergence
    n_jobs : int, default=None
        Number of parallel jobs to run
    cpu : bool, default=False
        If True, use CPU only
    device : int, default=0
        CUDA device index
    dtype : torch.dtype, default=torch.float64
        Data type for computations
    random_state : int, default=42
        Random seed for reproducibility

    Attributes:
    -----------
    embedding_ : Tensor of shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space
    stress_ : float
        The final value of the stress (disparity) of the embedding
    dissimilarity_matrix_ : Tensor of shape (n_samples, n_samples)
        Pairwise dissimilarity matrix of the training data
    n_iter_ : int
        Number of iterations run
    """

    def __init__(
        self,
        n_components: int,
        metric: bool = True,
        dissimilarity: str = "euclidean",
        p: int = 3,
        n_init: int = 4,
        max_iter: int = 300,
        eps: float = 1e-3,
        n_jobs: Optional[int] = None,
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
        self.metric = metric
        self.dissimilarity = dissimilarity
        self.p = p
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.n_jobs = n_jobs
        self.embedding_ = None
        self.stress_ = None
        self.dissimilarity_matrix_ = None
        self.n_iter_ = 0

    def __str__(self) -> str:
        return "MDS"

    def _compute_distance_matrix(self, X: Tensor) -> Tensor:
        """Compute pairwise distance matrix"""
        n_samples = X.shape[0]
        D = torch.zeros((n_samples, n_samples), device=self.device, dtype=self.dtype)

        if self.dissimilarity == "euclidean":
            calculate_method = euclidean_distance
        elif self.dissimilarity == "manhattan":
            calculate_method = manhattan_distance
        elif self.dissimilarity == "chebyshev":
            calculate_method = chebyshev_distance
        elif self.dissimilarity == "minkowski":
            calculate_method = minkowski_distance
        else:
            raise ValueError(f"Unknown dissimilarity measure: {self.dissimilarity}")

        if self.dissimilarity == "minkowski":
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    D[i, j] = calculate_method(
                        X[i].unsqueeze(0), X[j].unsqueeze(0), p=self.p
                    )
                    D[j, i] = D[i, j]
        else:
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    D[i, j] = calculate_method(X[i].unsqueeze(0), X[j].unsqueeze(0))
                    D[j, i] = D[i, j]

        return D

    def _double_center(self, B: Tensor) -> Tensor:
        """Double centering of a matrix"""
        n = B.shape[0]
        row_mean = torch.mean(B, dim=1, keepdim=True)
        col_mean = torch.mean(B, dim=0, keepdim=True)
        grand_mean = torch.mean(B)

        return B - row_mean - col_mean + grand_mean

    def _smacof_single(
        self, dissimilarities: Tensor, init: Tensor = None
    ) -> Tuple[Tensor, float, int]:
        """SMACOF algorithm for a single run"""
        n_samples = dissimilarities.shape[0]

        # Initialize the embedding
        if init is None:
            X = torch.randn(
                n_samples, self.n_components, device=self.device, dtype=self.dtype
            )
        else:
            X = init.clone()

        # Old stress value
        old_stress = None

        # SMACOF iterations
        for it in range(self.max_iter):
            # Compute distance matrix
            D = self._compute_distance_matrix(X)

            # Compute stress
            stress = torch.sqrt(torch.sum((dissimilarities - D) ** 2) / 2)

            # Check convergence
            if old_stress is not None and abs(old_stress - stress) < self.eps:
                break

            old_stress = stress

            # Compute scaling factors
            with torch.no_grad():
                # Avoid division by zero
                mask = D > 0
                scaling = torch.zeros_like(D)
                scaling[mask] = dissimilarities[mask] / D[mask]

                # Update embedding
                B = -scaling
                B.fill_diagonal_(0)
                row_sum = torch.sum(B, dim=1)
                for i in range(B.shape[0]):
                    B[i, i] = -row_sum[i]

                X = (B @ X) / n_samples

        return X, stress, it + 1

    def fit(self, X: Union[ndarray, Tensor]) -> "MDS":
        """Fit the MDS model with X"""
        X = self.check_input(X)
        n_samples = X.shape[0]

        # Compute dissimilarity matrix
        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X.clone()
        else:
            self.dissimilarity_matrix_ = self._compute_distance_matrix(X)

        # Ensure dissimilarities are symmetric and non-negative
        self.dissimilarity_matrix_ = (
            self.dissimilarity_matrix_ + self.dissimilarity_matrix_.T
        ) / 2
        self.dissimilarity_matrix_.clamp_min_(0)

        # Run SMACOF algorithm multiple times
        best_stress = float("inf")
        best_embedding = None
        best_n_iter = 0

        for i in range(self.n_init):
            # Initialize embedding
            init = torch.randn(
                n_samples, self.n_components, device=self.device, dtype=self.dtype
            )

            # Run SMACOF
            embedding, stress, n_iter = self._smacof_single(
                self.dissimilarity_matrix_, init
            )

            # Keep the best solution
            if stress < best_stress:
                best_stress = stress
                best_embedding = embedding
                best_n_iter = n_iter

        self.embedding_ = best_embedding
        self.stress_ = best_stress
        self.n_iter_ = best_n_iter

        return self

    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Transform X to the embedded space"""
        if self.embedding_ is None:
            raise RuntimeError("MDS must be fitted before transforming data")

        # For MDS, we can't transform new points directly
        # We need to recompute the embedding including the new points
        # This is a limitation of MDS
        raise NotImplementedError(
            "MDS does not support transforming new data. Use fit_transform instead."
        )

    def fit_transform(self, X: Union[ndarray, Tensor], *args) -> Union[ndarray, Tensor]:
        """Fit the model with X and apply the dimensionality reduction on X"""
        self.fit(X)
        return self.embedding_


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples, n_features = 100, 10
    X = torch.randn(n_samples, n_features, dtype=torch.float64)

    # Test MDS
    mds = MDS(n_components=2, n_init=4, max_iter=30)
    X_transformed = mds.fit_transform(X)

    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)
    print("Stress:", mds.stress_)
    print("Number of iterations:", mds.n_iter_)

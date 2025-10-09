import torch
from torch import Tensor

import numpy as np
from numpy import ndarray
from typing import Union, Optional

from models.base import BaseDimensionalityReduction
from utils.graph import knn_graph


class LLE(BaseDimensionalityReduction):
    """
    Locally Linear Embedding (LLE)

    A nonlinear dimensionality reduction technique that computes low-dimensional,
    neighborhood-preserving embeddings of high-dimensional inputs.

    Parameters:
    -----------
    n_components : int
        Number of coordinates for the manifold
    n_neighbors : int, default=5
        Number of neighbors to consider for each point
    reg : float, default=1e-3
        Regularization constant for the weight matrix
    eigen_solver : str, default='dense'
        Eigen solver to use ('dense', 'arpack')
    tol : float, default=1e-6
        Tolerance for iterative eigen solvers
    max_iter : int, default=100
        Maximum number of iterations for iterative eigen solvers
    method : str, default='standard'
        Method to use for LLE ('standard', 'hessian', 'modified', 'ltsa')
    distance_metric : str, default='euclidean'
        Distance metric to use for k-NN graph ('euclidean', 'manhattan', 'chebyshev', 'minkowski')
    p : int, default=3
        Power parameter for Minkowski distance
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
        Stores the embedding vectors
    reconstruction_error_ : float
        Reconstruction error associated with the embedding
    n_features_in_ : int
        Number of features seen during fit
    """

    def __init__(
        self,
        n_components: int,
        n_neighbors: int = 5,
        reg: float = 1e-3,
        eigen_solver: str = "dense",
        tol: float = 1e-6,
        max_iter: int = 100,
        method: str = "standard",
        distance_metric: str = "euclidean",
        p: int = 3,
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
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.distance_metric = distance_metric
        self.p = p
        self.embedding_ = None
        self.reconstruction_error_ = None
        self.n_features_in_ = None

        if self.eigen_solver not in ["dense", "arpack"]:
            raise ValueError(
                f"eigen_solver must be 'dense' or 'arpack', got {self.eigen_solver}"
            )

        if self.method not in ["standard", "hessian", "modified", "ltsa"]:
            raise ValueError(
                f"method must be 'standard', 'hessian', 'modified', or 'ltsa', got {self.method}"
            )

    def __str__(self) -> str:
        return "LLE"

    def _power_iteration(self, A, n_components, max_iter, tol):
        """Power iteration method for finding the smallest eigenvalues and eigenvectors"""
        n = A.shape[0]
        eigenvectors = torch.zeros(
            (n, n_components), device=self.device, dtype=self.dtype
        )
        eigenvalues = torch.zeros(n_components, device=self.device, dtype=self.dtype)

        # We want the smallest eigenvalues, so we use shifted inverse iteration
        # Shift by the largest eigenvalue estimate
        sigma = torch.trace(A) / n

        for i in range(n_components):
            # Initialize a random vector
            b = torch.randn(n, device=self.device, dtype=self.dtype)
            b = b / torch.norm(b)

            for _ in range(max_iter):
                # Solve (A - sigma*I) * b_new = b
                try:
                    # Try to use Cholesky decomposition for stability
                    L = torch.linalg.cholesky(
                        A - sigma * torch.eye(n, device=self.device, dtype=self.dtype)
                    )
                    b_new = torch.cholesky_solve(b.unsqueeze(1), L).squeeze()
                except:
                    # Fall back to LU decomposition
                    b_new = torch.linalg.solve(
                        A - sigma * torch.eye(n, device=self.device, dtype=self.dtype),
                        b,
                    )

                # Orthogonalize against previous eigenvectors
                for j in range(i):
                    b_new = (
                        b_new
                        - torch.dot(b_new, eigenvectors[:, j]) * eigenvectors[:, j]
                    )

                # Normalize
                b_new_norm = torch.norm(b_new)
                if b_new_norm < 1e-12:
                    break
                b_new = b_new / b_new_norm

                # Check convergence
                if torch.norm(b_new - b) < tol:
                    break

                b = b_new

            # Compute Rayleigh quotient (eigenvalue)
            eigenvalue = torch.dot(b, A @ b)

            # Store results
            eigenvalues[i] = eigenvalue
            eigenvectors[:, i] = b

            # Deflate matrix
            A = A - eigenvalue * torch.outer(b, b)

        return eigenvalues, eigenvectors

    def _compute_weights_standard(self, X: Tensor, indices: Tensor) -> Tensor:
        """Compute the reconstruction weights for standard LLE"""
        n_samples, n_features = X.shape
        n_neighbors = indices.shape[1]

        W = torch.zeros((n_samples, n_neighbors), device=self.device, dtype=self.dtype)

        for i in range(n_samples):
            neighbors = X[indices[i]]

            center = X[i]
            neighbors_centered = neighbors - center

            C = neighbors_centered @ neighbors_centered.T

            C += self.reg * torch.eye(n_neighbors, device=self.device, dtype=self.dtype)

            try:
                # Try to solve using Cholesky decomposition (more stable)
                L = torch.linalg.cholesky(C)
                w = torch.cholesky_solve(
                    torch.ones(n_neighbors, device=self.device, dtype=self.dtype), L
                )
            except:
                # Fall back to pseudo-inverse
                w = torch.linalg.pinv(C) @ torch.ones(
                    n_neighbors, device=self.device, dtype=self.dtype
                )

            w /= torch.sum(w)

            W[i] = w

        return W

    def _compute_weights_modified(self, X: Tensor, indices: Tensor) -> Tensor:
        """Compute the reconstruction weights for modified LLE"""
        n_samples, n_features = X.shape
        n_neighbors = indices.shape[1]

        W = torch.zeros((n_samples, n_neighbors), device=self.device, dtype=self.dtype)

        for i in range(n_samples):
            neighbors = X[indices[i]]

            center = X[i]
            neighbors_centered = neighbors - center

            C = neighbors_centered @ neighbors_centered.T

            # Modified LLE uses a different regularization approach
            trace_C = torch.trace(C)
            if trace_C > 0:
                C += (self.reg * trace_C / n_neighbors) * torch.eye(
                    n_neighbors, device=self.device, dtype=self.dtype
                )

            try:
                # Try to solve using Cholesky decomposition
                L = torch.linalg.cholesky(C)
                w = torch.cholesky_solve(
                    torch.ones(n_neighbors, device=self.device, dtype=self.dtype), L
                )
            except:
                # Fall back to pseudo-inverse
                w = torch.linalg.pinv(C) @ torch.ones(
                    n_neighbors, device=self.device, dtype=self.dtype
                )

            w /= torch.sum(w)

            W[i] = w

        return W

    def _compute_weights_hessian(self, X: Tensor, indices: Tensor) -> Tensor:
        """Compute the reconstruction weights for Hessian LLE"""
        n_samples, n_features = X.shape
        n_neighbors = indices.shape[1]

        W = torch.zeros((n_samples, n_neighbors), device=self.device, dtype=self.dtype)

        for i in range(n_samples):
            neighbors = X[indices[i]]

            center = X[i]
            neighbors_centered = neighbors - center

            # For Hessian LLE, we need to compute the local tangent space
            # using PCA on the neighborhood
            U, S, Vt = torch.linalg.svd(neighbors_centered, full_matrices=False)

            # Keep only the d largest principal components
            d = min(n_features, n_neighbors - 1)
            U = U[:, :d]

            # Construct the Hessian estimator
            H = torch.zeros(
                (d * (d + 1) // 2, n_neighbors), device=self.device, dtype=self.dtype
            )

            idx = 0
            for j in range(d):
                for k in range(j, d):
                    H[idx] = U[:, j] * U[:, k]
                    idx += 1

            # Orthonormalize H using QR decomposition
            Q, R = torch.linalg.qr(H.T)
            H_ortho = Q.T

            # The weights are given by the null space of H_ortho
            # We take the last row of H_ortho as the weight vector
            w = H_ortho[-1]
            w /= torch.sum(w)

            W[i] = w

        return W

    def _compute_weights_ltsa(self, X: Tensor, indices: Tensor) -> Tensor:
        """Compute the reconstruction weights for LTSA"""
        n_samples, n_features = X.shape
        n_neighbors = indices.shape[1]

        W = torch.zeros((n_samples, n_neighbors), device=self.device, dtype=self.dtype)

        for i in range(n_samples):
            neighbors = X[indices[i]]

            center = X[i]
            neighbors_centered = neighbors - center

            # Compute local tangent space using PCA
            U, S, Vt = torch.linalg.svd(neighbors_centered, full_matrices=False)

            # Keep only the d largest principal components
            d = min(n_features, n_neighbors - 1)
            U = U[:, :d]

            # Construct the alignment matrix
            I = torch.eye(n_neighbors, device=self.device, dtype=self.dtype)
            ones = torch.ones(n_neighbors, device=self.device, dtype=self.dtype)
            P = I - torch.outer(ones, ones) / n_neighbors

            # The weights are given by the solution to the linear system
            # (U.T @ P @ U) w = ones
            try:
                A = U.T @ P @ U
                L = torch.linalg.cholesky(A)
                w = torch.cholesky_solve(ones[:d], L)
            except:
                # Fall back to pseudo-inverse
                w = torch.linalg.pinv(U.T @ P @ U) @ ones[:d]

            # Transform back to the original space
            w = U @ w
            w /= torch.sum(w)

            W[i] = w

        return W

    def _compute_embedding(self, W: Tensor, method: str) -> Tensor:
        """Compute the LLE embedding from the weight matrix"""
        n_samples = W.shape[0]
        n_neighbors = W.shape[1]

        # Construct the sparse weight matrix
        W_sparse = torch.zeros(
            (n_samples, n_samples), device=self.device, dtype=self.dtype
        )

        # Create indices for the sparse matrix
        row_indices = torch.arange(n_samples, device=self.device).repeat_interleave(
            n_neighbors
        )
        col_indices = (
            torch.arange(n_samples, device=self.device)
            .repeat(n_neighbors)
            .reshape(n_neighbors, n_samples)
            .T.reshape(-1)
        )

        # Fill the sparse matrix
        W_sparse[row_indices, col_indices] = W.reshape(-1)

        # Construct the cost matrix based on the method
        if method == "standard":
            # Standard LLE: M = (I - W)^T (I - W)
            I = torch.eye(n_samples, device=self.device, dtype=self.dtype)
            M = (I - W_sparse).T @ (I - W_sparse)
        elif method == "modified":
            # Modified LLE: same as standard
            I = torch.eye(n_samples, device=self.device, dtype=self.dtype)
            M = (I - W_sparse).T @ (I - W_sparse)
        elif method == "hessian":
            # Hessian LLE: M = W^T W
            M = W_sparse.T @ W_sparse
        elif method == "ltsa":
            # LTSA: M = (I - W)^T (I - W)
            I = torch.eye(n_samples, device=self.device, dtype=self.dtype)
            M = (I - W_sparse).T @ (I - W_sparse)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute the bottom (n_components + 1) eigenvectors of M
        if self.eigen_solver == "dense":
            # Use dense solver
            eigenvalues, eigenvectors = torch.linalg.eigh(M)

            # Sort eigenvalues and eigenvectors in ascending order
            idx = torch.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        elif self.eigen_solver == "arpack":
            # Use power iteration method for smallest eigenvalues
            eigenvalues, eigenvectors = self._power_iteration(
                M, self.n_components + 1, self.max_iter, self.tol
            )

            # Sort in ascending order
            idx = torch.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        else:
            raise ValueError(f"Unsupported eigen_solver: {self.eigen_solver}")

        # Keep bottom n_components + 1 eigenvectors (excluding the first one)
        embedding = eigenvectors[:, 1 : self.n_components + 1]

        return embedding

    def fit(self, X: Union[ndarray, Tensor]) -> "LLE":
        """Fit the LLE model with X"""
        X = self.check_input(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        knn_dists, knn_indices = knn_graph(
            X, self.n_neighbors, metric=self.distance_metric, p=self.p
        )

        # Compute reconstruction weights based on method
        if self.method == "standard":
            W = self._compute_weights_standard(X, knn_indices)
        elif self.method == "modified":
            W = self._compute_weights_modified(X, knn_indices)
        elif self.method == "hessian":
            W = self._compute_weights_hessian(X, knn_indices)
        elif self.method == "ltsa":
            W = self._compute_weights_ltsa(X, knn_indices)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute embedding
        self.embedding_ = self._compute_embedding(W, self.method)

        # Compute reconstruction error
        self.reconstruction_error_ = torch.norm(
            X - self.embedding_ @ self.embedding_.T @ X
        )

        return self

    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Transform X to the embedded space"""
        if self.embedding_ is None:
            raise RuntimeError("LLE must be fitted before transforming data")

        # For LLE, we can't transform new points directly
        # We need to recompute the embedding including the new points
        # This is a limitation of LLE
        raise NotImplementedError(
            "LLE does not support transforming new data. Use fit_transform instead."
        )

    def fit_transform(self, X: Union[ndarray, Tensor], *args) -> Union[ndarray, Tensor]:
        """Fit the model with X and apply the dimensionality reduction on X"""
        self.fit(X)
        return self.embedding_


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples = 100

    # Generate Swiss roll
    t = 3 * np.pi / 2 * (1 + 2 * torch.rand(n_samples, dtype=torch.float64))
    x = t * torch.cos(t)
    y = 30 * torch.rand(n_samples, dtype=torch.float64)
    z = t * torch.sin(t)

    X = torch.stack([x, y, z], dim=1)

    # Test LLE with different methods
    methods = ["standard", "modified", "hessian", "ltsa"]

    for method in methods:
        try:
            lle = LLE(n_components=2, n_neighbors=12, method=method)
            X_transformed = lle.fit_transform(X)

            print(
                f"{method} LLE - Original shape: {X.shape}, Transformed shape: {X_transformed.shape}"
            )
            print(f"{method} LLE - Reconstruction error: {lle.reconstruction_error_}")
        except Exception as e:
            print(f"Error with {method} LLE: {e}")

import torch
from torch import Tensor

import numpy as np
from numpy import ndarray
from typing import Union, Optional, Callable, Tuple

from models.base import BaseDimensionalityReduction


class ICA(BaseDimensionalityReduction):
    """
    Independent Component Analysis (ICA)

    A computational method for separating a multivariate signal into additive subcomponents
    that are statistically independent from each other.

    Parameters:
    -----------
    n_components : int
        Number of independent components to extract
    algorithm : str, default='fastica'
        Algorithm to use for ICA ('fastica', 'infomax')
    fun : str or callable, default='logcosh' ('logcosh', 'exp', 'cube')
        Functional form of the G function used in the approximation to neg-entropy
    max_iter : int, default=200
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    whiten : bool, default=True
        Whether to whiten the data
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
    components_ : Tensor of shape (n_components, n_features)
        The unmixing matrix
    mixing_ : Tensor of shape (n_features, n_components)
        The mixing matrix
    mean_ : Tensor of shape (n_features,)
        The mean of the training data
    n_iter_ : int
        Number of iterations run
    """

    def __init__(
        self,
        n_components: int,
        algorithm: str = "fastica",
        fun: Union[str, Callable] = "logcosh",
        max_iter: int = 200,
        tol: float = 1e-4,
        whiten: bool = True,
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
        self.algorithm = algorithm
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.whiten = whiten
        self.mixing_ = None
        self.n_iter_ = 0

    def __str__(self) -> str:
        return "ICA"

    def _logcosh(self, x: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Logcosh function and its derivative"""
        gx = torch.tanh(alpha * x)
        g_x = alpha * (1 - gx**2)
        return gx, g_x

    def _exp(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Exp function and its derivative"""
        exp_x = torch.exp(-(x**2) / 2)
        gx = x * exp_x
        g_x = (1 - x**2) * exp_x
        return gx, g_x

    def _cube(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Cube function and its derivative"""
        return x**3, 3 * x**2

    def _center(self, X: Tensor) -> Tensor:
        """Center the data"""
        self.mean_ = torch.mean(X, dim=0)
        return X - self.mean_

    def _whiten(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Whiten the data using PCA"""
        n_samples, n_features = X.shape

        # Compute covariance matrix
        cov = X.T @ X / n_samples

        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep only n_components
        eigenvalues = eigenvalues[: self.n_components]
        eigenvectors = eigenvectors[:, : self.n_components]

        # Compute whitening matrix
        D = torch.diag(1.0 / torch.sqrt(eigenvalues))
        whitening_matrix = eigenvectors @ D

        # Whiten the data
        X_white = X @ whitening_matrix

        return X_white, whitening_matrix

    def _fastica(self, X: Tensor) -> Tensor:
        """FastICA algorithm"""
        n_samples, n_features = X.shape

        # Initialize random weights
        W = torch.randn(
            self.n_components, self.n_components, device=self.device, dtype=self.dtype
        )

        # Choose nonlinearity function
        if callable(self.fun):
            g = self.fun
        elif self.fun == "logcosh":
            g = self._logcosh
        elif self.fun == "exp":
            g = self._exp
        elif self.fun == "cube":
            g = self._cube
        else:
            raise ValueError(f"Unknown function: {self.fun}")

        # Symmetric decorrelation
        def symmetric_decorrelation(W):
            # W = (W * W.T)^{-1/2} * W
            s, u = torch.linalg.eigh(W @ W.T)
            return u @ torch.diag(1.0 / torch.sqrt(s)) @ u.T @ W

        # Main FastICA loop
        for i in range(self.max_iter):
            # Store previous W for convergence check
            W_prev = W.clone()

            # Compute new W
            wx = W @ X.T
            gwx, g_wx = g(wx)
            W_new = gwx @ X / n_samples - torch.diag(g_wx.mean(dim=1)) @ W

            # Symmetric decorrelation
            W = symmetric_decorrelation(W_new)

            # Check convergence
            lim = max(abs(abs(torch.diag(W @ W_prev.T)) - 1))
            if lim < self.tol:
                break

        self.n_iter_ = i + 1

        return W

    def _infomax(self, X: Tensor) -> Tensor:
        """Infomax algorithm"""
        n_samples, n_features = X.shape

        # Initialize random weights
        W = torch.eye(self.n_components, device=self.device, dtype=self.dtype)

        # Learning rate
        lr = 0.01 / torch.log(torch.tensor(self.n_components + 1))

        # Main Infomax loop
        for i in range(self.max_iter):
            # Store previous W for convergence check
            W_prev = W.clone()

            # Compute output
            y = torch.tanh(W @ X.T)

            # Compute gradient
            phi = 1 - 2 * y
            grad = (
                torch.eye(self.n_components, device=self.device, dtype=self.dtype)
                + phi @ y.T / n_samples
            )

            # Update weights
            W += lr * grad @ W

            # Check convergence
            if torch.norm(W - W_prev) < self.tol:
                break

        self.n_iter_ = i + 1

        return W

    def fit(self, X: Union[ndarray, Tensor]) -> "ICA":
        """Fit the ICA model with X"""
        X = self.check_input(X)
        n_samples, n_features = X.shape

        X_centered = self._center(X)

        # Whiten the data if requested
        if self.whiten:
            X_white, self.whitening_matrix_ = self._whiten(X_centered)
        else:
            X_white = X_centered
            self.whitening_matrix_ = torch.eye(
                n_features, device=self.device, dtype=self.dtype
            )

        # Run ICA algorithm
        if self.algorithm == "fastica":
            W = self._fastica(X_white)
        elif self.algorithm == "infomax":
            W = self._infomax(X_white)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Compute components
        self.components_ = W
        self.mixing_ = torch.pinverse(W)

        return self

    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Transform X to the independent components space"""
        if self.components_ is None:
            raise RuntimeError("ICA must be fitted before transforming data")

        X = self.check_input(X)
        X_centered = X - self.mean_

        # Whiten the data if requested
        if self.whiten:
            X_white = X_centered @ self.whitening_matrix_
        else:
            X_white = X_centered

        # Project to independent components space
        ica_scores = X_white @ self.components_.T

        return ica_scores

    def inverse_transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Transform independent components back to original space"""
        if self.components_ is None:
            raise RuntimeError("ICA must be fitted before inverse transforming data")

        X = self.check_input(X)

        # Reconstruct from independent components
        X_reconstructed = X @ torch.pinverse(self.components_)

        # Reverse whitening if applied
        if self.whiten:
            X_reconstructed = X_reconstructed @ torch.pinverse(self.whitening_matrix_)

        # Add mean back
        X_reconstructed += self.mean_

        return X_reconstructed


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples, n_features = 1000, 100

    t = torch.linspace(0, 10, n_samples)
    s1 = torch.sin(2 * t)
    s2 = torch.sign(torch.sin(3 * t))
    s3 = torch.tensor(np.random.laplace(size=n_samples))

    S = torch.stack([s1, s2, s3], dim=1)
    A = torch.randn(n_features, 3, dtype=torch.float64)
    X = S @ A.T

    # Test ICA
    ica = ICA(n_components=10, algorithm="fastica", fun="logcosh")
    X_transformed = ica.fit_transform(X)

    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)
    print("Number of iterations:", ica.n_iter_)

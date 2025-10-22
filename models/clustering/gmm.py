import torch
from torch import Tensor

from numpy import ndarray
from typing import Union, Tuple
import math

from models.base import BaseClustering
from models.clustering.kmeans import KMeans


class GaussianMixture(BaseClustering):
    """Gaussian Mixture Model implementation using PyTorch"""

    def __init__(
        self,
        n_clusters: int,
        max_iters: int = 100,
        tol: float = 1e-3,
        init_params: str = "kmeans",
        covariance_type: str = "full",      # TODO
        reg_covar: float = 1e-6,
        distance: str = "euclidean",
        p: int = 3,
        cpu: bool = False,
        device: int = 0,
        dtype: torch.dtype = torch.float64,
        random_state: int = 42,
    ):
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            distance=distance,
            p=p,
            cpu=cpu,
            device=device,
            dtype=dtype,
            random_state=random_state,
        )
        super().__init__(
            n_clusters=n_clusters,
            distance=distance,
            cpu=cpu,
            device=device,
            dtype=dtype,
            random_state=random_state,
        )
        self.max_iters = max_iters
        self.tol = tol
        self.init_params = init_params
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.p = p
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.lower_bounds_ = []

    def __str__(self):
        return "GaussianMixture"

    def _initialize_parameters(self, X: Tensor):
        """Initialize GMM parameters"""
        n_samples, n_features = X.shape

        if self.init_params == "kmeans":
            self.kmeans.fit(X)

            self.means_ = self.kmeans.cluster_centers_
            self.weights_ = (
                torch.ones(self.n_clusters, device=self.device) / self.n_clusters
            )

            # Initialize covariances
            self.covariances_ = torch.zeros(
                (self.n_clusters, n_features, n_features),
                device=self.device,
                dtype=self.dtype,
            )
            for k in range(self.n_clusters):
                cluster_points = X[self.kmeans.labels_ == k]
                if len(cluster_points) > 1:
                    centered = cluster_points - self.means_[k]
                    self.covariances_[k] = centered.T @ centered / len(cluster_points)
                else:
                    self.covariances_[k] = (
                        torch.eye(n_features, device=self.device) * 0.1
                    )

        elif self.init_params == "random":
            # Random initialization
            self.weights_ = (
                torch.ones(self.n_clusters, device=self.device) / self.n_clusters
            )
            indices = torch.randperm(n_samples, device=self.device)[: self.n_clusters]
            self.means_ = X[indices]

            self.covariances_ = torch.zeros(
                (self.n_clusters, n_features, n_features),
                device=self.device,
                dtype=self.dtype,
            )
            for k in range(self.n_clusters):
                self.covariances_[k] = torch.eye(n_features, device=self.device)

        else:
            raise ValueError(f"Unknown initialization method: {self.init_params}")

        # Add regularization to covariance matrices
        for k in range(self.n_clusters):
            self.covariances_[k] += (
                torch.eye(n_features, device=self.device) * self.reg_covar
            )

    def _multivariate_normal_logpdf(
        self, X: Tensor, mean: Tensor, cov: Tensor
    ) -> Tensor:
        """Compute log PDF of multivariate normal distribution"""
        n_features = X.shape[1]

        # Compute determinant and inverse of covariance matrix
        try:
            L = torch.linalg.cholesky(cov)
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
            Linv = torch.linalg.solve_triangular(
                L, torch.eye(n_features, device=self.device), upper=False
            )
            cov_inv = Linv.T @ Linv
        except RuntimeError:
            # If Cholesky fails, use SVD
            U, S, V = torch.svd(cov)
            log_det = torch.sum(torch.log(S))
            cov_inv = U @ torch.diag(1.0 / S) @ V.T

        # Compute quadratic form
        X_centered = X - mean
        quad_form = torch.sum(X_centered @ cov_inv * X_centered, dim=1)

        # Compute log PDF
        log_pdf = -0.5 * (n_features * math.log(2 * math.pi) + log_det + quad_form)

        return log_pdf

    def _e_step(self, X: Tensor) -> Tuple[Tensor, float]:
        """Expectation step: compute responsibilities"""
        n_samples = X.shape[0]
        log_resp = torch.zeros(
            (n_samples, self.n_clusters), device=self.device, dtype=self.dtype
        )

        # Compute log probabilities for each component
        for k in range(self.n_clusters):
            log_resp[:, k] = torch.log(
                self.weights_[k]
            ) + self._multivariate_normal_logpdf(
                X, self.means_[k], self.covariances_[k]
            )

        # Compute log sum exp for numerical stability
        log_sum_exp = torch.logsumexp(log_resp, dim=1, keepdim=True)
        log_resp_normalized = log_resp - log_sum_exp
        resp = torch.exp(log_resp_normalized)

        # Compute lower bound (evidence lower bound - ELBO)
        lower_bound = torch.sum(log_sum_exp)

        return resp, lower_bound.item()

    def _m_step(self, X: Tensor, resp: Tensor):
        """Maximization step: update parameters"""
        n_samples, n_features = X.shape

        # Update weights
        Nk = torch.sum(resp, dim=0)
        self.weights_ = Nk / n_samples

        # Update means
        self.means_ = (resp.T @ X) / Nk.unsqueeze(1)

        # Update covariances
        for k in range(self.n_clusters):
            centered = X - self.means_[k]
            weighted_centered = resp[:, k].unsqueeze(1) * centered
            self.covariances_[k] = (weighted_centered.T @ centered) / Nk[k]

            # Add regularization
            self.covariances_[k] += (
                torch.eye(n_features, device=self.device) * self.reg_covar
            )

    def fit(self, X: Union[ndarray, Tensor]) -> "GaussianMixture":
        """Fit GMM to the data using EM algorithm"""
        X_tensor = self._check_input(X)

        self._initialize_parameters(X_tensor)

        # EM algorithm
        prev_lower_bound = -float("inf")

        for iteration in range(self.max_iters):
            # E-step
            resp, lower_bound = self._e_step(X_tensor)
            self.lower_bounds_.append(lower_bound)

            # Check convergence
            if abs(lower_bound - prev_lower_bound) < self.tol:
                break

            # M-step
            self._m_step(X_tensor, resp)
            prev_lower_bound = lower_bound

        _, labels = torch.max(resp, dim=1)
        self.labels_ = labels
        self.cluster_centers_ = self.means_

        return self

    def predict(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Predict cluster labels for new data"""
        if self.means_ is None:
            raise ValueError("Model must be fitted before prediction")

        X_tensor = self._check_input(X)
        resp, _ = self._e_step(X_tensor)
        _, labels = torch.max(resp, dim=1)

        if isinstance(X, ndarray):
            return self.tensor2ndarray(labels)
        return labels

    def predict_proba(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Predict posterior probabilities for new data"""
        if self.means_ is None:
            raise ValueError("Model must be fitted before prediction")

        X_tensor = self._check_input(X)
        resp, _ = self._e_step(X_tensor)

        if isinstance(X, ndarray):
            return self.tensor2ndarray(resp)
        return resp


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples = 300
    n_features = 2

    cluster1 = torch.randn(n_samples // 3, n_features) + torch.tensor([2.0, 2.0])
    cluster2 = torch.randn(n_samples // 3, n_features) + torch.tensor([-2.0, -2.0])
    cluster3 = torch.randn(n_samples // 3, n_features) + torch.tensor([2.0, -2.0])
    X = torch.cat([cluster1, cluster2, cluster3], dim=0)

    # Fit GMM
    gmm = GaussianMixture(n_clusters=3, random_state=42)
    gmm.fit(X)

    print(f"GMM completed with {len(gmm.labels_)} samples")
    print(f"Means shape: {gmm.means_.shape}")
    print(f"Weights: {gmm.weights_}")
    print(f"Number of EM iterations: {len(gmm.lower_bounds_)}")

    # Test prediction
    test_points = torch.tensor([[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])
    predictions = gmm.predict(test_points)
    probabilities = gmm.predict_proba(test_points)

    print(f"Predictions for test points: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")

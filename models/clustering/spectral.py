import torch
from torch import Tensor

from numpy import ndarray
from typing import Union

from models.base import BaseClustering
from models.clustering.kmeans import KMeans
from utils.kernels import rbf_kernel
from utils.graph import knn_graph
from utils.distance import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
)


class SpectralClustering(BaseClustering):
    """Spectral Clustering algorithm implementation using PyTorch"""

    def __init__(
        self,
        n_clusters: int,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        eigen_solver: str = "svd",
        random_walk: bool = False,
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
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.eigen_solver = eigen_solver
        self.random_walk = random_walk
        self.p = p
        self.affinity_matrix_ = None
        self.eigenvectors_ = None

        if self.distance == "euclidean":
            self.calculate_method = euclidean_distance
        elif self.distance == "manhattan":
            self.calculate_method = manhattan_distance
        elif self.distance == "chebyshev":
            self.calculate_method = chebyshev_distance
        elif self.distance == "minkowski":
            self.calculate_method = minkowski_distance
        else:
            raise ValueError(f"Unknown distance measure: {self.distance}")

    def __str__(self):
        return "SpectralClustering"

    def _compute_affinity_matrix(self, X: Tensor) -> Tensor:
        """Compute affinity matrix based on specified method"""
        n_samples = X.shape[0]

        if self.affinity == "rbf":
            affinity_matrix = rbf_kernel(X, X, gamma=self.gamma)

        elif self.affinity == "nearest_neighbors":
            _, indices = knn_graph(
                X, k=self.n_neighbors, metric=self.distance, p=self.p
            )

            affinity_matrix = torch.zeros(
                (n_samples, n_samples), device=self.device, dtype=self.dtype
            )

            for i in range(n_samples):
                affinity_matrix[i, indices[i]] = 1.0
                affinity_matrix[indices[i], i] = 1.0

            # Make it symmetric
            affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2

        elif self.affinity == "precomputed":
            affinity_matrix = X

        else:
            raise ValueError(f"Unsupported affinity: {self.affinity}")

        return affinity_matrix

    def _compute_laplacian(self, affinity_matrix: Tensor) -> Tensor:
        """Compute graph Laplacian"""
        degree_matrix = torch.diag(torch.sum(affinity_matrix, dim=1))

        if self.random_walk:
            # Random walk normalized Laplacian: L_rw = I - D ^ {-1} W
            degree_inv = torch.diag(1.0 / torch.diag(degree_matrix))
            laplacian = (
                torch.eye(affinity_matrix.shape[0], device=self.device)
                - degree_inv @ affinity_matrix
            )
        else:
            # Symmetric normalized Laplacian: L_sym = I - D ^ {-1/2} W D ^ {-1/2}
            degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(torch.diag(degree_matrix)))
            laplacian = (
                torch.eye(affinity_matrix.shape[0], device=self.device)
                - degree_sqrt_inv @ affinity_matrix @ degree_sqrt_inv
            )

        return laplacian

    def _compute_eigenvectors(self, laplacian: Tensor) -> Tensor:
        """Compute eigenvectors of Laplacian matrix"""
        if self.eigen_solver == "svd":
            U, S, Vt = torch.svd(laplacian)
            eigenvectors = U
        elif self.eigen_solver == "eig":
            eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        else:
            raise ValueError(f"Unsupported eigen solver: {self.eigen_solver}")

        return eigenvectors

    def fit(self, X: Union[ndarray, Tensor]) -> "SpectralClustering":
        """Fit Spectral Clustering to the data"""
        X_tensor = self._check_input(X)

        self.affinity_matrix_ = self._compute_affinity_matrix(X_tensor)

        laplacian = self._compute_laplacian(self.affinity_matrix_)

        eigenvectors = self._compute_eigenvectors(laplacian)

        # Select the k smallest eigenvectors (excluding the first for normalized Laplacian)
        if self.random_walk:
            # For random walk Laplacian, use first k eigenvectors
            embedding = eigenvectors[:, : self.n_clusters]
        else:
            # For symmetric Laplacian, use eigenvectors corresponding to smallest eigenvalues
            embedding = eigenvectors[
                :, 1 : self.n_clusters + 1
            ]  # Skip first trivial eigenvector

        # Normalize rows to unit length (for k-means)
        row_norms = torch.norm(embedding, p=2, dim=1, keepdim=True)
        embedding_normalized = embedding / row_norms

        self.eigenvectors_ = embedding_normalized

        self.kmeans.fit(embedding_normalized)

        self.labels_ = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_

        return self

    def predict(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Predict cluster labels for new data"""
        if self.affinity_matrix_ is None:
            raise ValueError("Model must be fitted before prediction")

        # For spectral clustering, prediction is non-trivial
        # We'll use the approach of finding nearest neighbors in the training set
        X_tensor = self._check_input(X)
        n_samples = X_tensor.shape[0]

        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)

        for i in range(n_samples):
            if self.distance == "minkowski":
                distances = self.calculate_method(
                    self.affinity_matrix_, X_tensor[i].unsqueeze(0), p=self.p
                )
            else:
                distances = self.calculate_method(
                    self.affinity_matrix_, X_tensor[i].unsqueeze(0)
                )
            min_idx = torch.argmin(distances)
            labels[i] = self.labels_[min_idx]

        if isinstance(X, ndarray):
            return self.tensor2ndarray(labels)
        return labels


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples = 300

    theta = torch.linspace(0, 2 * torch.pi, n_samples // 2)
    r1 = torch.ones(n_samples // 4) * 2
    r2 = torch.ones(n_samples // 4) * 4

    # Inner circle
    x1 = r1 * torch.cos(theta[: n_samples // 4])
    y1 = r1 * torch.sin(theta[: n_samples // 4])
    inner_circle = torch.stack([x1, y1], dim=1)

    # Outer circle
    x2 = r2 * torch.cos(theta[: n_samples // 4])
    y2 = r2 * torch.sin(theta[: n_samples // 4])
    outer_circle = torch.stack([x2, y2], dim=1)

    X = torch.cat([inner_circle, outer_circle], dim=0)
    X += torch.randn_like(X) * 0.1

    # Fit Spectral Clustering
    spectral = SpectralClustering(n_clusters=2, affinity="rbf", gamma=1.0)
    spectral.fit(X)

    print(f"Spectral Clustering completed with {len(spectral.labels_)} samples")
    print(f"Affinity matrix shape: {spectral.affinity_matrix_.shape}")
    print(f"Eigenvectors shape: {spectral.eigenvectors_.shape}")

    unique_labels = torch.unique(spectral.labels_)
    print(f"Unique labels: {unique_labels}")

    for label in unique_labels:
        count = torch.sum(spectral.labels_ == label).item()
        print(f"Cluster {label}: {count} points")

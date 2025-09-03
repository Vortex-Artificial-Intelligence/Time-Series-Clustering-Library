import torch
from torch import Tensor

import numpy as np
from numpy import ndarray
from typing import Union, Optional

from models.base import BaseDimensionalityReduction
from utils.graph import knn_graph


class Isomap(BaseDimensionalityReduction):
    """
    Isomap (Isometric Mapping)
    
    A nonlinear dimensionality reduction method that extends MDS by using geodesic distances
    instead of Euclidean distances to preserve the intrinsic geometry of the data.
    
    Parameters:
    -----------
    n_components : int
        Number of coordinates for the manifold
    n_neighbors : int, default=5
        Number of neighbors to consider for each point
    distance_metric : str, default='euclidean'
        Distance metric to use for k-NN graph ('euclidean', 'manhattan', 'chebyshev', 'minkowski')
    p : int, default=3
        Power parameter for Minkowski distance
    path_method : str, default='FW'
        Method to use for finding shortest paths ('FW', 'D')
    eigen_solver : str, default='dense'
        Eigen solver to use ('dense')    # ToDo : 'arpack'
    tol : float, default=0    # ToDo
        Convergence tolerance passed to arpack or lobpcg
    max_iter : int, default=None    # ToDo
        Maximum number of iterations for the arpack solver
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
    kernel_pca_ : object
        KernelPCA object used for the embedding
    nbrs_ : NearestNeighbors instance
        Stores nearest neighbors instance
    dist_matrix_ : Tensor of shape (n_samples, n_samples)
        Stores the geodesic distance matrix of the training data
    n_features_in_ : int
        Number of features seen during fit
    """
    
    def __init__(
        self,
        n_components: int,
        n_neighbors: int = 5,
        distance_metric: str = "euclidean",
        p: int = 3,
        path_method: str = "FW",
        eigen_solver: str = "dense",
        tol: float = 0,
        max_iter: Optional[int] = None,
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
        self.distance_metric = distance_metric
        self.p = p
        self.path_method = path_method
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.embedding_ = None
        self.dist_matrix_ = None
        self.n_features_in_ = None
        
    def __str__(self) -> str:
        return "Isomap"
    
    def _compute_geodesic_distances(self, X: Tensor) -> Tensor:
        """Compute geodesic distances using k-NN graph and shortest path algorithm"""
        n_samples = X.shape[0]

        knn_dists, knn_indices = knn_graph(X, self.n_neighbors, metric=self.distance_metric, p=self.p)

        D = torch.full((n_samples, n_samples), float('inf'), device=self.device, dtype=self.dtype)
        torch.diagonal(D).fill_(0)

        # Fill initial distances from KNN
        for i in range(n_samples):
            for j in range(self.n_neighbors):
                neighbor_idx = knn_indices[i, j]
                D[i, neighbor_idx] = knn_dists[i, j]
                D[neighbor_idx, i] = knn_dists[i, j]  # Assuming symmetric

        # Compute shortest paths
        if self.path_method == "D":
            # Dijkstra's algorithm
            for i in range(n_samples):
                # Initialize distances and visited set for this source
                dist = torch.full((n_samples,), float('inf'), device=self.device, dtype=self.dtype)
                dist[i] = 0
                visited = torch.zeros(n_samples, dtype=torch.bool, device=self.device)

                for _ in range(n_samples):
                    # Find the unvisited node with the smallest distance
                    min_dist = float('inf')
                    min_index = -1
                    for j in range(n_samples):
                        if not visited[j] and dist[j] < min_dist:
                            min_dist = dist[j]
                            min_index = j

                    if min_index == -1:  # All remaining nodes are unreachable
                        break
                    
                    visited[min_index] = True

                    # Update distances to neighbors
                    for j in range(n_samples):
                        if not visited[j] and D[min_index, j] < float('inf'):
                            new_dist = dist[min_index] + D[min_index, j]
                            if new_dist < dist[j]:
                                dist[j] = new_dist

                # Update the distance matrix with results from this source
                for j in range(n_samples):
                    if dist[j] < D[i, j]:  # Only update if we found a shorter path
                        D[i, j] = dist[j]
        elif self.path_method == "FW":
            # Use Floyd-Warshall algorithm
            for k in range(n_samples):
                for i in range(n_samples):
                    if D[i, k] == float('inf'):
                        continue
                    for j in range(n_samples):
                        if D[i, j] > D[i, k] + D[k, j]:
                            D[i, j] = D[i, k] + D[k, j]
        else:
            raise ValueError(f"Unknown path_method: {self.path_method}")

        return D
    
    def _isomap_embedding(self, D: Tensor) -> Tensor:
        """Compute the Isomap embedding from the geodesic distance matrix"""
        n_samples = D.shape[0]
        
        # Center the distance matrix
        H = torch.eye(n_samples, device=self.device, dtype=self.dtype) - torch.ones((n_samples, n_samples), device=self.device, dtype=self.dtype) / n_samples
        K = -0.5 * H @ D**2 @ H
        
        # Eigen decomposition (use dense)
        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top n_components
        eigenvalues = eigenvalues[:self.n_components]
        eigenvectors = eigenvectors[:, :self.n_components]
        
        # Compute embedding
        embedding = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))
        
        return embedding
    
    def fit(self, X: Union[ndarray, Tensor]) -> "Isomap":
        """Fit the Isomap model with X"""
        X = self.check_input(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.dist_matrix_ = self._compute_geodesic_distances(X)

        self.embedding_ = self._isomap_embedding(self.dist_matrix_)
        
        return self
    
    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Transform X to the embedded space"""
        if self.embedding_ is None:
            raise RuntimeError("Isomap must be fitted before transforming data")
            
        # For Isomap, we can't transform new points directly
        # We need to recompute the embedding including the new points
        # This is a limitation of Isomap
        raise NotImplementedError("Isomap does not support transforming new data. Use fit_transform instead.")
    
    def fit_transform(self, X: Union[ndarray, Tensor], *args) -> Union[ndarray, Tensor]:
        """Fit the model with X and apply the dimensionality reduction on X"""
        self.fit(X)
        return self.embedding_


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples = 100

    t = 3 * np.pi / 2 * (1 + 2 * torch.rand(n_samples, dtype=torch.float64))
    x = t * torch.cos(t)
    y = 30 * torch.rand(n_samples, dtype=torch.float64)
    z = t * torch.sin(t)
    
    X = torch.stack([x, y, z], dim=1)
    
    # Test Isomap
    isomap = Isomap(n_components=2, n_neighbors=10)
    X_transformed = isomap.fit_transform(X)
    
    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)

import torch
from torch import Tensor

from numpy import ndarray
from typing import Union, List

from models.base import BaseClustering
from utils.distance import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
)


class DBSCAN(BaseClustering):
    """DBSCAN clustering algorithm implementation using PyTorch"""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        distance: str = "euclidean",
        p: int = 3,
        cpu: bool = False,
        device: int = 0,
        dtype: torch.dtype = torch.float64,
        random_state: int = 42,
    ):
        super().__init__(
            n_clusters=None,  # DBSCAN doesn't require n_clusters
            distance=distance,
            cpu=cpu,
            device=device,
            dtype=dtype,
            random_state=random_state,
        )
        self.eps = eps
        self.min_samples = min_samples
        self.p = p
        self.core_sample_indices_ = None
        self.components_ = None

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
        return "DBSCAN"

    def _region_query(self, X: Tensor, point_idx: int) -> List[int]:
        """Find all points within eps distance of point"""
        if self.distance == "minkowski":
            distances = self.calculate_method(X, X[point_idx].unsqueeze(0), p=self.p)
        else:
            distances = self.calculate_method(X, X[point_idx].unsqueeze(0))
        neighbors = torch.where(distances <= self.eps)[0]
        return neighbors.tolist()

    def _expand_cluster(
        self,
        X: Tensor,
        labels: List[int],
        point_idx: int,
        neighbors: List[int],
        cluster_id: int,
    ) -> bool:
        """Expand cluster from core point"""
        labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:  # If neighbor is noise
                labels[neighbor_idx] = cluster_id

                # Find neighbors of neighbor
                neighbor_neighbors = self._region_query(X, neighbor_idx)
                if len(neighbor_neighbors) >= self.min_samples:
                    # Add new neighbors to the list
                    for new_neighbor in neighbor_neighbors:
                        if new_neighbor not in neighbors:
                            neighbors.append(new_neighbor)

            i += 1

        return True

    def fit(self, X: Union[ndarray, Tensor]) -> "DBSCAN":
        """Fit DBSCAN to the data"""
        X_tensor = self._check_input(X)
        n_samples = X_tensor.shape[0]

        # Initialize labels: 0 = unvisited, -1 = noise, >0 = cluster ID
        labels = [0] * n_samples  # Start with all unvisited
        cluster_id = 0

        core_samples = []

        for point_idx in range(n_samples):
            if labels[point_idx] != 0:  # Already visited
                continue

            # Find neighbors
            neighbors = self._region_query(X_tensor, point_idx)

            if len(neighbors) < self.min_samples:
                # Mark as noise
                labels[point_idx] = -1
            else:
                # Start new cluster
                cluster_id += 1
                core_samples.append(point_idx)
                self._expand_cluster(X_tensor, labels, point_idx, neighbors, cluster_id)

        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, device=self.device, dtype=torch.long)

        self.labels_ = labels_tensor
        self.core_sample_indices_ = torch.tensor(
            core_samples, device=self.device, dtype=torch.long
        )
        self.components_ = X_tensor[self.core_sample_indices_]
        self.n_clusters = cluster_id

        return self

    def predict(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Predict cluster labels for new data (nearest core point)"""
        if self.core_sample_indices_ is None:
            raise ValueError("Model must be fitted before prediction")

        X_tensor = self._check_input(X)
        n_samples = X_tensor.shape[0]

        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)

        for i in range(n_samples):
            if self.distance == "minkowski":
                distances = self.calculate_method(
                    self.components_, X_tensor[i].unsqueeze(0), p=self.p
                )
            else:
                distances = self.calculate_method(
                    self.components_, X_tensor[i].unsqueeze(0)
                )
            min_idx = torch.argmin(distances)
            labels[i] = self.labels_[self.core_sample_indices_[min_idx]]

        if isinstance(X, ndarray):
            return self.tensor2ndarray(labels)
        return labels


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples = 200

    cluster1 = torch.randn(n_samples // 2, 2) + torch.tensor([3.0, 3.0])
    cluster2 = torch.randn(n_samples // 2, 2) + torch.tensor([-3.0, -3.0])
    noise = torch.randn(20, 2) * 5
    X = torch.cat([cluster1, cluster2, noise], dim=0)

    # Fit DBSCAN
    dbscan = DBSCAN(eps=1.0, min_samples=5, random_state=42)
    dbscan.fit(X)

    print(f"DBSCAN completed with {len(dbscan.labels_)} samples")
    print(f"Number of clusters found: {dbscan.n_clusters}")
    print(f"Number of core samples: {len(dbscan.core_sample_indices_)}")

    unique_labels = torch.unique(dbscan.labels_)
    print(f"Unique labels: {unique_labels}")

    # Count points in each cluster and noise
    for label in unique_labels:
        if label == -1:
            print(f"Noise points: {torch.sum(dbscan.labels_ == label).item()}")
        else:
            count = torch.sum(dbscan.labels_ == label).item()
            print(f"Cluster {label}: {count} points")

    # Test prediction
    test_points = torch.tensor([[3.0, 3.0], [-3.0, -3.0], [0.0, 0.0]])
    predictions = dbscan.predict(test_points)
    print(f"Predictions for test points: {predictions}")

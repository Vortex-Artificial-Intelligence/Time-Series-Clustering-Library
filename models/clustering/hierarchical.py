import torch
from torch import Tensor

from numpy import ndarray
from typing import Union, Tuple, Dict, List, Any
import heapq

from models.base import BaseClustering
from utils.distance import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
)


class HierarchicalClustering(BaseClustering):
    """Hierarchical Agglomerative Clustering implementation using PyTorch"""

    def __init__(
        self,
        n_clusters: int,
        linkage: str = "ward",
        distance: str = "euclidean",
        p: int = 3,
        compute_full_tree: bool = False,        # TODO
        cpu: bool = False,
        device: int = 0,
        dtype: torch.dtype = torch.float64,
        random_state: int = 42,
    ):
        super().__init__(
            n_clusters=n_clusters,
            distance=distance,
            cpu=cpu,
            device=device,
            dtype=dtype,
            random_state=random_state,
        )
        self.linkage = linkage
        self.p = p
        self.compute_full_tree = compute_full_tree
        self.children_ = None
        self.n_leaves_ = None
        self.n_components_ = None
        self.X_fit_ = None

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
        return "HierarchicalClustering"

    def _compute_distance_matrix(self, X: Tensor) -> Tensor:
        """Compute pairwise distance matrix"""
        n_samples = X.shape[0]
        distance_matrix = torch.zeros(
            (n_samples, n_samples), device=self.device, dtype=self.dtype
        )

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if self.distance == "minkowski":
                    dist = self.calculate_method(
                        X[i].unsqueeze(0), X[j].unsqueeze(0), p=self.p
                    )
                else:
                    dist = self.calculate_method(X[i].unsqueeze(0), X[j].unsqueeze(0))
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def _compute_ward_distance(
        self, cluster_i: Tuple, cluster_j: Tuple, cluster_info: Dict
    ) -> float:
        """Compute Ward distance between two clusters"""
        n_i = cluster_info[cluster_i]["size"]
        n_j = cluster_info[cluster_j]["size"]

        # Ward's method: minimize variance increase
        centroid_i = cluster_info[cluster_i]["centroid"]
        centroid_j = cluster_info[cluster_j]["centroid"]

        if self.distance == "minkowski":
            dist = self.calculate_method(
                centroid_i.unsqueeze(0), centroid_j.unsqueeze(0), p=self.p
            )
        else:
            dist = self.calculate_method(
                centroid_i.unsqueeze(0), centroid_j.unsqueeze(0)
            )

        ward_dist = torch.sqrt((2 * n_i * n_j * dist**2) / (n_i + n_j))
        return ward_dist.item()

    def _compute_linkage_distance(
        self,
        cluster_i: Tuple,
        cluster_j: Tuple,
        distance_matrix: Tensor,
        cluster_info: Dict,
    ) -> float:
        """Compute linkage distance between two clusters"""
        indices_i = cluster_info[cluster_i]["indices"]
        indices_j = cluster_info[cluster_j]["indices"]

        if self.linkage == "single":
            # Single linkage: minimum distance
            min_dist = float("inf")
            for idx_i in indices_i:
                for idx_j in indices_j:
                    dist = distance_matrix[idx_i, idx_j].item()
                    if dist < min_dist:
                        min_dist = dist
            return min_dist

        elif self.linkage == "complete":
            # Complete linkage: maximum distance
            max_dist = 0
            for idx_i in indices_i:
                for idx_j in indices_j:
                    dist = distance_matrix[idx_i, idx_j].item()
                    if dist > max_dist:
                        max_dist = dist
            return max_dist

        elif self.linkage == "average":
            # Average linkage: average distance
            total_dist = 0
            count = 0
            for idx_i in indices_i:
                for idx_j in indices_j:
                    total_dist += distance_matrix[idx_i, idx_j].item()
                    count += 1
            return total_dist / count if count > 0 else float("inf")

        else:
            raise ValueError(f"Unsupported linkage: {self.linkage}")

    def fit(self, X: Union[ndarray, Tensor]) -> "HierarchicalClustering":
        """Fit Hierarchical Clustering to the data"""
        X_tensor = self._check_input(X)
        n_samples = X_tensor.shape[0]

        # Store training data for prediction
        self.X_fit_ = X_tensor.clone()

        # Initialize: each point is its own cluster
        # Use frozenset as cluster identifier to ensure hashability and uniqueness
        clusters: List[frozenset] = [frozenset([i]) for i in range(n_samples)]
        cluster_info: Dict[frozenset, Dict[str, Any]] = {}

        for i, cluster in enumerate(clusters):
            cluster_info[cluster] = {"indices": [i], "size": 1, "centroid": X_tensor[i]}

        if self.linkage != "ward":
            distance_matrix = self._compute_distance_matrix(X_tensor)

        # Priority queue for closest clusters
        heap = []

        # Initialize heap with all pairwise distances
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_i = clusters[i]
                cluster_j = clusters[j]

                if self.linkage == "ward":
                    dist = self._compute_ward_distance(
                        cluster_i, cluster_j, cluster_info
                    )
                else:
                    dist = self._compute_linkage_distance(
                        cluster_i, cluster_j, distance_matrix, cluster_info
                    )

                heapq.heappush(heap, (dist, cluster_i, cluster_j))

        # Agglomerative clustering
        children = []
        current_cluster_id = n_samples

        while len(clusters) > self.n_clusters:
            if not heap:
                break

            # Find closest clusters
            dist, cluster_i, cluster_j = heapq.heappop(heap)

            # Skip if clusters no longer exist
            if cluster_i not in cluster_info or cluster_j not in cluster_info:
                continue

            # Skip if clusters are not in the current clusters list
            if cluster_i not in clusters or cluster_j not in clusters:
                continue

            # Merge clusters
            new_cluster = cluster_i.union(cluster_j)

            # Update cluster info
            indices_i = cluster_info[cluster_i]["indices"]
            indices_j = cluster_info[cluster_j]["indices"]
            size_i = cluster_info[cluster_i]["size"]
            size_j = cluster_info[cluster_j]["size"]

            if self.linkage == "ward":
                # Update centroid for Ward's method
                centroid_i = cluster_info[cluster_i]["centroid"]
                centroid_j = cluster_info[cluster_j]["centroid"]
                new_centroid = (size_i * centroid_i + size_j * centroid_j) / (
                    size_i + size_j
                )
            else:
                new_centroid = None

            cluster_info[new_cluster] = {
                "indices": indices_i + indices_j,
                "size": size_i + size_j,
                "centroid": new_centroid,
            }

            # Record the merge
            children.append(
                (
                    current_cluster_id,
                    min(cluster_i),  # Use min element as representative
                    min(cluster_j),  # Use min element as representative
                    dist,
                )
            )

            # Remove old clusters
            clusters.remove(cluster_i)
            clusters.remove(cluster_j)
            clusters.append(new_cluster)

            # Remove old clusters from cluster_info
            del cluster_info[cluster_i]
            del cluster_info[cluster_j]

            # Update heap with new distances
            for other_cluster in clusters:
                if other_cluster != new_cluster:
                    if self.linkage == "ward":
                        dist = self._compute_ward_distance(
                            new_cluster, other_cluster, cluster_info
                        )
                    else:
                        dist = self._compute_linkage_distance(
                            new_cluster, other_cluster, distance_matrix, cluster_info
                        )

                    heapq.heappush(heap, (dist, new_cluster, other_cluster))

            current_cluster_id += 1

        # Build labels from final clusters
        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster_info[cluster]["indices"]:
                labels[idx] = cluster_id

        self.labels_ = labels
        self.children_ = children
        self.n_leaves_ = n_samples
        self.n_components_ = len(clusters)

        # Compute and store cluster centers for prediction
        self._compute_cluster_centers(X_tensor)

        return self

    def _compute_cluster_centers(self, X: Tensor) -> None:
        """Compute cluster centers from training data and labels"""
        unique_labels = torch.unique(self.labels_)
        cluster_centers = []

        for label in unique_labels:
            mask = self.labels_ == label
            if torch.sum(mask) > 0:
                centroid = torch.mean(X[mask], dim=0)
                cluster_centers.append(centroid)

        self.cluster_centers_ = (
            torch.stack(cluster_centers) if cluster_centers else None
        )

    def predict(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Predict cluster labels for new data (nearest cluster center)"""
        if self.children_ is None:
            raise ValueError("Model must be fitted before prediction")

        X_tensor = self._check_input(X)
        n_samples = X_tensor.shape[0]

        if self.cluster_centers_ is None:
            raise ValueError(
                "Cluster centers not available. Model may not be properly fitted."
            )

        centroids = self.cluster_centers_
        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)

        for i in range(n_samples):
            if self.distance == "minkowski":
                distances = self.calculate_method(
                    centroids, X_tensor[i].unsqueeze(0), p=self.p
                )
            else:
                distances = self.calculate_method(centroids, X_tensor[i].unsqueeze(0))
            min_idx = torch.argmin(distances)
            labels[i] = min_idx

        if isinstance(X, ndarray):
            return self.tensor2ndarray(labels)
        return labels


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples = 100
    n_features = 2

    cluster1 = torch.randn(n_samples // 3, n_features) + torch.tensor([3.0, 3.0])
    cluster2 = torch.randn(n_samples // 3, n_features) + torch.tensor([-3.0, -3.0])
    cluster3 = torch.randn(n_samples // 3, n_features) + torch.tensor([3.0, -3.0])
    X = torch.cat([cluster1, cluster2, cluster3], dim=0)

    # Test different linkages and distances
    test_cases = [
        ("ward", "euclidean"),
        ("single", "euclidean"),
        ("complete", "euclidean"),
        ("average", "euclidean"),
    ]

    for linkage, distance in test_cases:
        print(f"\nTesting {linkage} linkage with {distance} distance:")

        # Fit Hierarchical Clustering
        hierarchical = HierarchicalClustering(
            n_clusters=3, linkage=linkage, distance=distance
        )
        hierarchical.fit(X)

        print(f"  Completed with {len(hierarchical.labels_)} samples")
        print(f"  Number of clusters found: {hierarchical.n_components_}")
        print(f"  Number of merges: {len(hierarchical.children_)}")
        print(f"  Cluster centers shape: {hierarchical.cluster_centers_.shape}")

        unique_labels = torch.unique(hierarchical.labels_)
        print(f"  Unique labels: {unique_labels}")

        for label in unique_labels:
            count = torch.sum(hierarchical.labels_ == label).item()
            print(f"  Cluster {label}: {count} points")

        # Test prediction on new data
        test_points = torch.tensor([[3.0, 3.0], [-3.0, -3.0], [3.0, -3.0], [0.0, 0.0]])
        predictions = hierarchical.predict(test_points)
        print(f"  Predictions for test points: {predictions}")

        print(f"  {linkage} with {distance} test passed!")

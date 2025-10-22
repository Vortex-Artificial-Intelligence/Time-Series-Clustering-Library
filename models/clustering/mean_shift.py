import torch
from torch import Tensor

import numpy as np
from numpy import ndarray
from typing import Optional, Union, List, Tuple

from models.base import BaseClustering
from utils.distance import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
)


class MeanShift(BaseClustering):
    """Mean Shift clustering algorithm implementation using PyTorch"""
    
    def __init__(
        self,
        bandwidth: Optional[float] = None,
        max_iters: int = 300,
        tol: float = 1e-3,
        bin_seeding: bool = False,
        min_bin_freq: int = 1,
        distance: str = "euclidean",
        p: int = 3,
        cpu: bool = False,
        device: int = 0,
        dtype: torch.dtype = torch.float64,
        random_state: int = 42,
    ):
        super().__init__(
            n_clusters=None,    # Mean Shift doesn't require n_clusters
            distance=distance,
            cpu=cpu,
            device=device,
            dtype=dtype,
            random_state=random_state
        )
        self.bandwidth = bandwidth
        self.max_iters = max_iters
        self.tol = tol
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.p = p
        self.cluster_centers_ = None
        
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
        return "MeanShift"
    
    def _estimate_bandwidth(self, X: Tensor) -> float:
        """Estimate bandwidth using median heuristic"""
        n_samples = X.shape[0]
        
        if n_samples > 1000:
            indices = torch.randperm(n_samples, device=self.device)[:1000]
            X_subset = X[indices]
        else:
            X_subset = X
        
        distances = []
        for i in range(X_subset.shape[0]):
            for j in range(i + 1, X_subset.shape[0]):
                if self.distance == "minkowski":
                    dist = self.calculate_method(X_subset[i].unsqueeze(0), X_subset[j].unsqueeze(0), p=self.p)
                else:
                    dist = self.calculate_method(X_subset[i].unsqueeze(0), X_subset[j].unsqueeze(0))
                distances.append(dist.item())
        
        # Use median distance as bandwidth
        bandwidth = np.median(distances)
        return bandwidth
    
    def _get_bin_seeds(self, X: Tensor, bin_size: float) -> Tensor:
        """Get initial seeds using bin seeding"""
        n_samples, n_features = X.shape

        bins = {}
        bin_counts = {}
        
        for i in range(n_samples):
            # Compute bin coordinates
            bin_coords = tuple((X[i] / bin_size).floor().int().tolist())
            
            if bin_coords in bins:
                bins[bin_coords] += X[i]
                bin_counts[bin_coords] += 1
            else:
                bins[bin_coords] = X[i].clone()
                bin_counts[bin_coords] = 1
        
        # Collect seeds from bins with sufficient points
        seeds = []
        for bin_coords, count in bin_counts.items():
            if count >= self.min_bin_freq:
                seed = bins[bin_coords] / count
                seeds.append(seed)
        
        return torch.stack(seeds) if seeds else X[torch.randint(0, n_samples, (1,))]
    
    def _mean_shift_single(self, X: Tensor, start_point: Tensor) -> Tuple[Tensor, int]:
        """Perform mean shift for a single starting point"""
        point = start_point.clone()
        
        for iteration in range(self.max_iters):
            if self.distance == "minkowski":
               distances = self.calculate_method(X, point.unsqueeze(0), p=self.p)
            else:
                distances = self.calculate_method(X, point.unsqueeze(0))
            
            # Compute weights using Gaussian kernel
            weights = torch.exp(-0.5 * (distances / self.bandwidth)**2)
            
            # Compute weighted mean
            weighted_sum = torch.sum(weights.unsqueeze(1) * X, dim=0)
            total_weight = torch.sum(weights)
            
            if total_weight == 0:
                break
                
            new_point = weighted_sum / total_weight
            
            # Check convergence
            if self.distance == "minkowski":
                shift = self.calculate_method(new_point.unsqueeze(0), point.unsqueeze(0), p=self.p).item()
            else:
                shift = self.calculate_method(new_point.unsqueeze(0), point.unsqueeze(0)).item()
            point = new_point
            
            if shift < self.tol:
                break
        
        return point, iteration + 1
    
    def _merge_clusters(self, centers: List[Tensor], merge_threshold: float) -> Tuple[Tensor, List[int]]:
        """Merge nearby cluster centers"""
        if not centers:
            return torch.empty((0, 0), device=self.device), []
            
        centers_tensor = torch.stack(centers)
        n_centers = centers_tensor.shape[0]

        cluster_labels = list(range(n_centers))
        
        for i in range(n_centers):
            if cluster_labels[i] != i:
                continue
                
            for j in range(i + 1, n_centers):
                if cluster_labels[j] != j: 
                    continue
                    
                if self.distance == "minkowski":
                    dist = self.calculate_method(centers_tensor[i].unsqueeze(0), centers_tensor[j].unsqueeze(0), p=self.p).item()
                else:
                    dist = self.calculate_method(centers_tensor[i].unsqueeze(0), centers_tensor[j].unsqueeze(0)).item()
                
                if dist < merge_threshold:
                    cluster_labels[j] = i

        unique_labels = []
        final_centers = []
        label_map = {}
        
        for i in range(n_centers):
            root_label = cluster_labels[i]
            while cluster_labels[root_label] != root_label:
                root_label = cluster_labels[root_label]
                
            if root_label not in label_map:
                label_map[root_label] = len(final_centers)
                final_centers.append(centers_tensor[root_label])
                unique_labels.append([])
                
            unique_labels[label_map[root_label]].append(i)
        
        return torch.stack(final_centers), unique_labels
    
    def fit(self, X: Union[ndarray, Tensor]) -> 'MeanShift':
        """Fit Mean Shift to the data"""
        X_tensor = self._check_input(X)
        n_samples, n_features = X_tensor.shape

        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(X_tensor)
            print(f"Estimated bandwidth: {self.bandwidth:.4f}")

        if self.bin_seeding:
            bin_size = self.bandwidth
            seeds = self._get_bin_seeds(X_tensor, bin_size)
        else:
            seeds = X_tensor

        all_centers = []
        convergence_iters = []
        
        for i in range(seeds.shape[0]):
            center, iters = self._mean_shift_single(X_tensor, seeds[i])
            all_centers.append(center)
            convergence_iters.append(iters)

        merge_threshold = self.bandwidth * 0.5  # Merge centers within half bandwidth
        cluster_centers, cluster_members = self._merge_clusters(all_centers, merge_threshold)
        
        # Assign points to clusters
        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        
        for i in range(n_samples):
            if self.distance == "minkowski":
                distances = self.calculate_method(cluster_centers, X_tensor[i].unsqueeze(0), p=self.p)
            else:
                distances = self.calculate_method(cluster_centers, X_tensor[i].unsqueeze(0))
            min_idx = torch.argmin(distances)
            labels[i] = min_idx

        self.labels_ = labels
        self.cluster_centers_ = cluster_centers
        self.n_clusters = cluster_centers.shape[0]
        
        return self
    
    def predict(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Predict cluster labels for new data"""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
            
        X_tensor = self._check_input(X)
        n_samples = X_tensor.shape[0]
        
        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        
        for i in range(n_samples):
            if self.distance == "minkowski":
                distances = self.calculate_method(self.cluster_centers_, X_tensor[i].unsqueeze(0), p=self.p)
            else:
                distances = self.calculate_method(self.cluster_centers_, X_tensor[i].unsqueeze(0))
            min_idx = torch.argmin(distances)
            labels[i] = min_idx
        
        if isinstance(X, ndarray):
            return self.tensor2ndarray(labels)
        return labels


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples = 200

    cluster1 = torch.randn(n_samples//3, 2) + torch.tensor([3.0, 3.0])
    cluster2 = torch.randn(n_samples//3, 2) + torch.tensor([-3.0, -3.0])
    cluster3 = torch.randn(n_samples//3, 2) + torch.tensor([3.0, -3.0])
    X = torch.cat([cluster1, cluster2, cluster3], dim=0)
    
    # Fit Mean Shift
    mean_shift = MeanShift(bandwidth=2.0)
    mean_shift.fit(X)

    print(f"MeanShift completed with {len(mean_shift.labels_)} samples")
    print(f"Number of clusters found: {mean_shift.n_clusters}")
    print(f"Cluster centers shape: {mean_shift.cluster_centers_.shape}")
    
    unique_labels = torch.unique(mean_shift.labels_)
    print(f"Unique labels: {unique_labels}")
    
    for label in unique_labels:
        count = torch.sum(mean_shift.labels_ == label).item()
        print(f"Cluster {label}: {count} points")
    
    # Test prediction
    test_points = torch.tensor([[3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]])
    predictions = mean_shift.predict(test_points)
    print(f"Predictions for test points: {predictions}")

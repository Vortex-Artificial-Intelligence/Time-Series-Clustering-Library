import torch
from torch import Tensor

import numpy as np
from numpy import ndarray

from typing import Optional, Union, Tuple

from models.base import BaseClustering
from utils.distance import euclidean_distance


class KMeans(BaseClustering):
    """K-Means clustering algorithm implementation using PyTorch"""
    
    def __init__(
        self,
        n_clusters: int,
        max_iters: int = 300,
        tol: float = 1e-4,
        init: str = 'k-means++',
        distance: str = "euclidean",
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
            random_state=random_state
        )
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.inertia_ = None
        
    def __str__(self):
        return "KMeans"
    
    def _initialize_centroids(self, X: Tensor) -> Tensor:
        """Initialize cluster centroids using specified method"""
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Random initialization
            indices = torch.randperm(n_samples, device=self.device)[:self.n_clusters]
            centroids = X[indices]
            
        elif self.init == 'k-means++':
            # K-means++ initialization
            centroids = torch.zeros((self.n_clusters, n_features), 
                                  device=self.device, dtype=self.dtype)
            
            # First centroid: choose randomly
            first_idx = torch.randint(0, n_samples, (1,), device=self.device)
            centroids[0] = X[first_idx]
            
            # Subsequent centroids: choose with probability proportional to distance^2
            for i in range(1, self.n_clusters):
                # Compute distances to nearest centroid for all points
                distances = self._compute_min_distances(X, centroids[:i])
                
                # Choose next centroid with probability proportional to distance^2
                probabilities = distances**2 / torch.sum(distances**2)
                cumulative_probs = torch.cumsum(probabilities, dim=0)
                r = torch.rand(1, device=self.device)
                next_idx = torch.searchsorted(cumulative_probs, r)
                centroids[i] = X[next_idx]
                
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
            
        return centroids
    
    def _compute_min_distances(self, X: Tensor, centroids: Tensor) -> Tensor:
        """Compute minimum distance from each point to any centroid"""
        n_samples = X.shape[0]
        min_distances = torch.full((n_samples,), float('inf'), 
                                 device=self.device, dtype=self.dtype)
        
        for centroid in centroids:
            distances = euclidean_distance(X, centroid.unsqueeze(0))
            min_distances = torch.min(min_distances, distances)
            
        return min_distances
    
    def _assign_clusters(self, X: Tensor, centroids: Tensor) -> Tuple[Tensor, Tensor]:
        """Assign points to nearest clusters and compute inertia"""
        n_samples = X.shape[0]
        labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)
        inertia = 0.0
        
        for i in range(n_samples):
            distances = torch.zeros(self.n_clusters, device=self.device, dtype=self.dtype)
            for j in range(self.n_clusters):
                distances[j] = euclidean_distance(X[i].unsqueeze(0), centroids[j].unsqueeze(0))
            
            min_idx = torch.argmin(distances)
            labels[i] = min_idx
            inertia += distances[min_idx].item()
            
        return labels, inertia
    
    def _update_centroids(self, X: Tensor, labels: Tensor) -> Tensor:
        """Update centroids based on current cluster assignments"""
        new_centroids = torch.zeros((self.n_clusters, X.shape[1]), 
                                  device=self.device, dtype=self.dtype)
        
        for i in range(self.n_clusters):
            mask = labels == i
            if torch.sum(mask) > 0:
                new_centroids[i] = torch.mean(X[mask], dim=0)
            else:
                # Handle empty clusters by reinitializing
                new_centroids[i] = X[torch.randint(0, X.shape[0], (1,), device=self.device)]
                
        return new_centroids
    
    def fit(self, X: Union[ndarray, Tensor]) -> 'KMeans':
        """Fit K-Means model to the data"""
        X_tensor = self._check_input(X)
        
        # Initialize centroids
        centroids = self._initialize_centroids(X_tensor)
        
        # Main iteration loop
        for iteration in range(self.max_iters):
            # Assign points to clusters
            labels, inertia = self._assign_clusters(X_tensor, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X_tensor, labels)
            
            # Check convergence
            centroid_shift = torch.sqrt(torch.sum((new_centroids - centroids)**2, dim=1)).max()
            
            centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
                
        # Store results
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        
        return self
    
    def predict(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Predict cluster labels for new data"""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
            
        X_tensor = self._check_input(X)
        labels, _ = self._assign_clusters(X_tensor, self.cluster_centers_)
        
        if isinstance(X, ndarray):
            return self.tensor2ndarray(labels)
        return labels


# Test function
def test_kmeans():
    """Test KMeans implementation"""
    print("Testing KMeans...")
    
    # Generate sample data
    torch.manual_seed(42)
    n_samples = 300
    n_features = 2
    
    # Create three clusters
    cluster1 = torch.randn(n_samples//3, n_features) + torch.tensor([2.0, 2.0])
    cluster2 = torch.randn(n_samples//3, n_features) + torch.tensor([-2.0, -2.0])
    cluster3 = torch.randn(n_samples//3, n_features) + torch.tensor([2.0, -2.0])
    
    X = torch.cat([cluster1, cluster2, cluster3], dim=0)
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Check results
    print(f"KMeans completed in {len(kmeans.labels_)} samples")
    print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
    print(f"Inertia: {kmeans.inertia_:.4f}")
    
    # Test prediction
    test_points = torch.tensor([[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])
    predictions = kmeans.predict(test_points)
    print(f"Predictions for test points: {predictions}")
    
    print("KMeans test passed!\n")


if __name__ == "__main__":
    test_kmeans()


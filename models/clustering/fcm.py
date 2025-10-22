import torch
from torch import Tensor

from numpy import ndarray
from typing import Union

from models.base import BaseClustering
from utils.distance import (
    euclidean_distance,
    manhattan_distance,
    chebyshev_distance,
    minkowski_distance,
)


class FuzzyCMeans(BaseClustering):
    """Fuzzy C-Means clustering algorithm implementation using PyTorch"""
    
    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        max_iters: int = 100,
        tol: float = 1e-4,
        distance: str = "euclidean",
        p: int = 3,
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
        self.m = m
        self.max_iters = max_iters
        self.tol = tol
        self.p = p
        self.membership_ = None
        self.inertia_ = None
        self.history_ = []
    
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
        return "FuzzyCMeans"
    
    def _initialize_membership(self, n_samples: int) -> Tensor:
        """Initialize membership matrix randomly"""
        membership = torch.rand(n_samples, self.n_clusters, 
                              device=self.device, dtype=self.dtype)
        row_sums = torch.sum(membership, dim=1, keepdim=True)
        membership = membership / row_sums
        return membership
    
    def _update_centers(self, X: Tensor, membership: Tensor) -> Tensor:
        """Update cluster centers based on current membership"""
        membership_power = membership ** self.m
        weighted_sum = membership_power.T @ X
        weights_sum = torch.sum(membership_power, dim=0, keepdim=True).T
        weights_sum = torch.where(weights_sum == 0, torch.tensor(1e-8, device=self.device), weights_sum)
        centers = weighted_sum / weights_sum
        return centers
    
    def _update_membership(self, X: Tensor, centers: Tensor) -> Tensor:
        """Update membership matrix based on current centers"""
        n_samples = X.shape[0]

        distances = torch.zeros((n_samples, self.n_clusters), 
                              device=self.device, dtype=self.dtype)
        
        for j in range(self.n_clusters):
            if self.distance == "minkowski":
                distances[:, j] = self.calculate_method(X, centers[j].unsqueeze(0), p=self.p)
            else:
                distances[:, j] = self.calculate_method(X, centers[j].unsqueeze(0))

        distances = torch.where(distances == 0, torch.tensor(1e-8, device=self.device), distances)

        exponent = 2.0 / (self.m - 1.0)

        membership = torch.zeros((n_samples, self.n_clusters), 
                               device=self.device, dtype=self.dtype)
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                ratio_sum = 0.0
                for k in range(self.n_clusters):
                    ratio = distances[i, j] / distances[i, k]
                    ratio_sum += ratio ** exponent
                membership[i, j] = 1.0 / ratio_sum
        
        return membership
    
    def _compute_objective(self, X: Tensor, centers: Tensor, membership: Tensor) -> float:
        """Compute FCM objective function value"""
        objective = 0.0
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                if self.distance == "minkowski":
                    dist = self.calculate_method(X[i].unsqueeze(0), centers[j].unsqueeze(0), p=self.p)
                else:
                    dist = self.calculate_method(X[i].unsqueeze(0), centers[j].unsqueeze(0))
                objective += (membership[i, j] ** self.m) * (dist ** 2)
        
        return objective.item()
    
    def fit(self, X: Union[ndarray, Tensor]) -> 'FuzzyCMeans':
        """Fit FCM to the data"""
        X_tensor = self._check_input(X)
        n_samples, n_features = X_tensor.shape

        membership = self._initialize_membership(n_samples)

        prev_objective = float('inf')
        
        for iteration in range(self.max_iters):
            centers = self._update_centers(X_tensor, membership)
            new_membership = self._update_membership(X_tensor, centers)
            objective = self._compute_objective(X_tensor, centers, new_membership)
            self.history_.append(objective)
        
            # Check convergence
            if abs(objective - prev_objective) < self.tol:
                break
                
            membership = new_membership
            prev_objective = objective

        self.membership_ = membership
        self.cluster_centers_ = centers
        self.inertia_ = objective

        _, labels = torch.max(membership, dim=1)
        self.labels_ = labels
        
        return self
    
    def predict(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Predict cluster labels for new data"""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
            
        X_tensor = self._check_input(X)

        membership = self._update_membership(X_tensor, self.cluster_centers_)

        _, labels = torch.max(membership, dim=1)
        
        if isinstance(X, ndarray):
            return self.tensor2ndarray(labels)
        return labels
    
    def predict_proba(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Predict membership probabilities for new data"""
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
            
        X_tensor = self._check_input(X)
        membership = self._update_membership(X_tensor, self.cluster_centers_)
        
        if isinstance(X, ndarray):
            return self.tensor2ndarray(membership)
        return membership
    
    def get_membership(self) -> Union[ndarray, Tensor]:
        """Get membership matrix for training data"""
        if self.membership_ is None:
            raise ValueError("Model must be fitted before getting membership")
        
        if self.inputs_type == ndarray:
            return self.tensor2ndarray(self.membership_)
        return self.membership_


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples = 300
    n_features = 2

    cluster1 = torch.randn(n_samples//3, n_features) + torch.tensor([2.0, 2.0])
    cluster2 = torch.randn(n_samples//3, n_features) + torch.tensor([-2.0, -2.0])
    cluster3 = torch.randn(n_samples//3, n_features) + torch.tensor([2.0, -2.0])
    
    X = torch.cat([cluster1, cluster2, cluster3], dim=0)
    
    # Fit FCM
    fcm = FuzzyCMeans(n_clusters=3, m=2.0)
    fcm.fit(X)

    print(f"FCM completed with {len(fcm.labels_)} samples")
    print(f"Cluster centers shape: {fcm.cluster_centers_.shape}")
    print(f"Membership matrix shape: {fcm.membership_.shape}")
    print(f"Final objective value: {fcm.inertia_:.4f}")
    print(f"Number of iterations: {len(fcm.history_)}")
    
    # Check membership properties
    membership_sum = torch.sum(fcm.membership_, dim=1)
    print(f"Membership sums (should be close to 1): min={membership_sum.min().item():.4f}, "
          f"max={membership_sum.max().item():.4f}")
    
    unique_labels = torch.unique(fcm.labels_)
    print(f"Unique labels: {unique_labels}")
    
    for label in unique_labels:
        count = torch.sum(fcm.labels_ == label).item()
        print(f"Cluster {label}: {count} points")
    
    # Test prediction
    test_points = torch.tensor([[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [0.0, 0.0]])
    
    # Test hard prediction
    predictions = fcm.predict(test_points)
    print(f"Hard predictions for test points: {predictions}")
    
    # Test soft prediction (membership probabilities)
    probabilities = fcm.predict_proba(test_points)
    print(f"Membership probabilities shape: {probabilities.shape}")
    print(f"Sample membership probabilities:\n{probabilities[:2]}")
    
    # Verify that membership probabilities sum to 1 for each point
    prob_sums = torch.sum(probabilities, dim=1)
    print(f"Probability sums: {prob_sums}")

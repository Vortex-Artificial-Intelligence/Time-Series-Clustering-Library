import torch
from torch import Tensor

import numpy as np
from numpy import ndarray
from typing import Union, Optional

from models.base import BaseDimensionalityReduction
from utils.validation import check_array
from utils.decomposition import svd_flip


class IncrementalPCA(BaseDimensionalityReduction):
    """Incremental Principal Component Analysis (IncrementalPCA)"""
    
    def __init__(
        self,
        n_components: int,
        whiten: bool = False,
        batch_size: Optional[int] = None,
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
        self.whiten = whiten
        self.batch_size = batch_size
        self.singular_values_ = None
        self.var_ = None
        self.total_samples_seen_ = 0
        self.iterated = False
        
    def __str__(self) -> str:
        return "IncrementalPCA"
    
    def fit(self, X: Union[ndarray, Tensor]) -> "IncrementalPCA":
        """Fit the model with X"""
        X = check_array(X, dtype=self.dtype, device=self.device)
        n_samples, n_features = X.shape
        
        if self.batch_size is None:
            self.batch_size = 5 * n_features if n_features > 0 else 100
        
        # Initialize components if not already done
        if self.components_ is None:
            self.components_ = torch.zeros(
                (self.n_components, n_features), 
                dtype=self.dtype, 
                device=self.device
            )
            
        if self.mean_ is None:
            self.mean_ = torch.zeros(n_features, dtype=self.dtype, device=self.device)
            
        if self.var_ is None:
            self.var_ = torch.zeros(n_features, dtype=self.dtype, device=self.device)
            
        # Process data in batches
        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i + self.batch_size]
            self.partial_fit(batch)
            
        return self
    
    def partial_fit(self, X: Union[ndarray, Tensor]) -> "IncrementalPCA":
        """Incremental fit with X"""
        X = check_array(X, dtype=self.dtype, device=self.device)
        n_samples, n_features = X.shape
        
        if self.components_ is None:
            self.components_ = torch.zeros(
                (self.n_components, n_features), 
                dtype=self.dtype, 
                device=self.device
            )
            
        if self.mean_ is None:
            self.mean_ = torch.zeros(n_features, dtype=self.dtype, device=self.device)
            
        if self.var_ is None:
            self.var_ = torch.zeros(n_features, dtype=self.dtype, device=self.device)
            
        # Update mean and variance
        col_mean = torch.mean(X, dim=0)
        col_var = torch.var(X, dim=0, unbiased=False)
        
        # First pass
        if not self.iterated:
            self.mean_ = col_mean
            self.var_ = col_var
            self.total_samples_seen_ = n_samples
            self.iterated = True
        else:
            # Update mean and variance
            total_samples = self.total_samples_seen_ + n_samples
            mean_old = self.mean_.clone()
            self.mean_ = (self.total_samples_seen_ * self.mean_ + n_samples * col_mean) / total_samples
            
            # Update variance
            self.var_ = (
                (self.total_samples_seen_ * self.var_ + 
                 n_samples * col_var + 
                 self.total_samples_seen_ * (mean_old - self.mean_)**2 +
                 n_samples * (col_mean - self.mean_)**2) / total_samples
            )
            
            self.total_samples_seen_ = total_samples
        
        # Center data
        X_centered = X - self.mean_
        
        # Update components using SVD on the concatenated matrix
        if self.total_samples_seen_ > self.n_components:
            if self.components_ is not None and torch.any(self.components_ != 0):
                # We already have components, update them
                U, S, Vt = torch.linalg.svd(
                    torch.cat([torch.diag(self.singular_values_) @ self.components_, X_centered], dim=0),
                    full_matrices=False
                )
            else:
                # First batch
                U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
                
            # Flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U, Vt)
            
            # Store components and singular values
            self.components_ = Vt[:self.n_components]
            self.singular_values_ = S[:self.n_components]
            
            # Compute explained variance
            self.explained_variance_ = (S**2) / (self.total_samples_seen_ - 1)
            self.explained_variance_ratio_ = self.explained_variance_ / torch.sum(self.var_)
            
        return self
    
    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Apply dimensionality reduction to X"""
        if self.components_ is None:
            raise RuntimeError("IncrementalPCA must be fitted before transforming data")
            
        X = check_array(X, dtype=self.dtype, device=self.device)
        X_centered = X - self.mean_
        
        # Project data
        X_transformed = X_centered @ self.components_.T
        
        # Whitening if required
        if self.whiten:
            X_transformed /= torch.sqrt(self.explained_variance_[:self.n_components])
            
        return X_transformed


# Test
if __name__ == "__main__":
    # Generate sample data
    torch.manual_seed(42)
    X = torch.randn(10, 5)
    
    # Test IncrementalPCA
    ipca = IncrementalPCA(n_components=2, batch_size=20)
    X_transformed = ipca.fit_transform(X)
    
    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)
    print("Explained variance ratio:", ipca.explained_variance_ratio_)
    
    # Test partial fit
    X_new = torch.randn(10, 5)
    ipca.partial_fit(X_new)
    X_new_transformed = ipca.transform(X_new)
    print("New data transformed shape:", X_new_transformed.shape)

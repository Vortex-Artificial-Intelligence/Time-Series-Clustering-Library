import torch
from torch import Tensor

import numpy as np
from numpy import ndarray
from typing import Optional, Union
import warnings

from models.base import BaseDimensionalityReduction
from utils.validation import check_array


class SparsePCA(BaseDimensionalityReduction):
    """Sparse Principal Component Analysis (SparsePCA)
    
    Finds the set of sparse components that can optimally reconstruct the data.
    The amount of sparseness is controllable by the parameter alpha.
    
    Parameters
    ----------
    n_components : int
        Number of sparse components to extract.
        
    alpha : float, default=1
        Sparsity controlling parameter. Higher values lead to sparser components.
        
    ridge_alpha : float, default=0.01
        Amount of ridge shrinkage to apply in order to improve conditioning.
        
    max_iter : int, default=1000
        Maximum number of iterations to perform.
        
    tol : float, default=1e-8
        Tolerance for the stopping condition.
        
    method : {'lars', 'cd'}, default='cd'
        Method to be used for optimization.
        lars: uses the least angle regression method (memory intensive)
        cd: uses coordinate descent (more memory efficient)
        
    n_jobs : int, default=None
        Number of parallel jobs to run.
        
    cpu : bool, default=False
        If True, use CPU only.
        
    device : int, default=0
        CUDA device index.
        
    dtype : torch.dtype, default=torch.float64
        Data type for tensors.
        
    random_state : int, default=42
        Seed for random number generation.
        
    Attributes
    ----------
    components_ : Tensor of shape (n_components, n_features)
        Sparse components extracted from the data.
        
    error_ : Tensor
        Reconstruction error at each iteration.
        
    n_iter_ : int
        Number of iterations run.
    """
    
    def __init__(
        self,
        n_components: int,
        alpha: float = 1,
        ridge_alpha: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        method: str = "cd",
        n_jobs: Optional[int] = None,
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
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_jobs = n_jobs
        self.error_ = None
        self.n_iter_ = 0
        
    def __str__(self) -> str:
        return "SparsePCA"
    
    def fit(self, X: Union[ndarray, Tensor]) -> "SparsePCA":
        """Fit the model from data in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, dtype=self.dtype, device=self.device)
        n_samples, n_features = X.shape
        
        # Center data
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        
        # Initialize components
        if self.components_ is None:
            try:
                # Try to use SVD for initialization
                U, S, Vt = torch.svd(X_centered)
                self.components_ = Vt[:self.n_components].clone().to(
                    dtype=self.dtype, device=self.device
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Fallback to random initialization if SVD fails
                warnings.warn("SVD initialization failed, using random initialization")
                torch.manual_seed(self.random_state)
                self.components_ = torch.randn(
                    self.n_components, n_features, 
                    dtype=self.dtype, device=self.device
                )
                # Normalize components
                norms = torch.norm(self.components_, dim=1, keepdim=True)
                self.components_ = self.components_ / torch.clamp(norms, min=1e-8)
        
        # Initialize optimization variables
        self.n_iter_ = 0
        self.error_ = torch.zeros(self.max_iter, device=self.device, dtype=self.dtype)
        
        # Alternate optimization
        for i in range(self.max_iter):
            try:
                # Update code
                code = self._update_code(X_centered)
                
                # Update components
                self.components_ = self._update_components(X_centered, code)
                
                # Calculate reconstruction error
                reconstruction = code @ self.components_
                self.error_[i] = torch.norm(X_centered - reconstruction, p="fro") ** 2
                
                # Check convergence
                if i > 0 and torch.abs(self.error_[i] - self.error_[i - 1]) < self.tol:
                    self.n_iter_ = i + 1
                    break
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                warnings.warn(f"Memory error at iteration {i}, stopping early")
                self.n_iter_ = i
                self.error_ = self.error_[:i]
                break
        else:
            self.n_iter_ = self.max_iter
            
        # Truncate error array
        self.error_ = self.error_[:self.n_iter_]
        
        return self
    
    def transform(self, X: Union[ndarray, Tensor]) -> Tensor:
        """Transform data to the sparse code representation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        code : Tensor of shape (n_samples, n_components)
            Transformed data.
        """
        X = check_array(X, dtype=self.dtype, device=self.device)
        X_centered = X - self.mean_
        return self._update_code(X_centered)
    
    def _update_code(self, X: Tensor) -> Tensor:
        """Update the sparse code.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Centered data.
            
        Returns
        -------
        code : Tensor of shape (n_samples, n_components)
            Updated sparse code.
        """
        if self.method == "lars":
            # LARS is more memory intensive, so we use a batch approach
            return self._lars_batch(X, self.components_, self.alpha)
        else:  # method == "cd"
            return self._coordinate_descent(X, self.components_, self.alpha)
    
    def _update_components(self, X: Tensor, code: Tensor) -> Tensor:
        """Update the sparse components.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Centered data.
            
        code : Tensor of shape (n_samples, n_components)
            Current sparse code.
            
        Returns
        -------
        components : Tensor of shape (n_components, n_features)
            Updated sparse components.
        """
        n_components = self.components_.shape[0]
        
        # Solve ridge regression problem with more memory-efficient approach
        # Use iterative method if direct solve fails
        try:
            A = code.T @ code + self.ridge_alpha * torch.eye(
                n_components, device=self.device, dtype=self.dtype
            )
            B = code.T @ X
            components = torch.linalg.solve(A, B)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Fallback to iterative method if direct solve fails
            warnings.warn("Direct solve failed, using iterative method")
            components = self._iterative_solve(code, X, n_components)
        
        # Normalize components
        norms = torch.norm(components, dim=1, keepdim=True)
        components = components / torch.clamp(norms, min=1e-8)
        
        return components
    
    def _iterative_solve(self, code: Tensor, X: Tensor, n_components: int) -> Tensor:
        """Iterative method to solve the ridge regression problem.
        
        Parameters
        ----------
        code : Tensor of shape (n_samples, n_components)
            Current sparse code.
            
        X : Tensor of shape (n_samples, n_features)
            Centered data.
            
        n_components : int
            Number of components.
            
        Returns
        -------
        components : Tensor of shape (n_components, n_features)
            Solution to the ridge regression problem.
        """
        # Use gradient descent to solve the ridge regression problem
        components = torch.zeros(
            n_components, X.shape[1], device=self.device, dtype=self.dtype
        )
        
        # Precompute code.T @ code
        A = code.T @ code
        B = code.T @ X
        
        # Gradient descent parameters
        lr = 0.01
        max_iter = 100
        tol = 1e-6
        
        for i in range(max_iter):
            # Compute gradient
            gradient = A @ components - B + self.ridge_alpha * components
            
            # Update components
            components = components - lr * gradient
            
            # Check convergence
            if torch.norm(gradient) < tol:
                break
                
        return components
    
    def _lars_batch(self, X: Tensor, components: Tensor, alpha: float) -> Tensor:
        """Batch version of Least Angle Regression for sparse coding.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to encode.
            
        components : Tensor of shape (n_components, n_features)
            Current components.
            
        alpha : float
            Sparsity controlling parameter.
            
        Returns
        -------
        code : Tensor of shape (n_samples, n_components)
            Sparse code.
        """
        n_samples, n_features = X.shape
        n_components = components.shape[0]
        
        # Process in batches to reduce memory usage
        batch_size = min(100, n_samples)  # Adjust batch size as needed
        code = torch.zeros(n_samples, n_components, device=self.device, dtype=self.dtype)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            
            # Process each sample in the batch
            for i in range(X_batch.shape[0]):
                # Initialize variables for LARS
                residual = X_batch[i].clone()  # Current residual
                active = torch.tensor([], dtype=torch.long, device=self.device)  # Active set
                signs = torch.tensor([], dtype=torch.long, device=self.device)  # Signs of active coefficients
                mu = torch.zeros(n_components, device=self.device, dtype=self.dtype)  # Current solution
                
                # LARS loop
                while True:
                    # Compute correlations
                    c = components @ residual
                    C = torch.max(torch.abs(c))
                    
                    # Check stopping condition
                    if C <= alpha:
                        break
                        
                    # Update active set
                    new_active = torch.where(torch.abs(c) >= C - 1e-10)[0]
                    active = torch.cat([active, new_active])
                    signs = torch.cat([signs, torch.sign(c[new_active]).long()])
                    
                    # Remove duplicates (if any)
                    active, unique_indices = torch.unique(active, return_inverse=True)
                    signs = signs[unique_indices]
                    
                    # Compute equiangular vector
                    GA = components[active] @ components[active].T
                    ones = torch.ones(len(active), device=self.device, dtype=self.dtype)
                    A = 1.0 / torch.sqrt(ones @ torch.linalg.solve(GA, ones))
                    w = A * torch.linalg.solve(GA, ones)
                    u = (components[active].T * signs).T @ w
                    
                    # Compute step size
                    gamma = torch.tensor(float("inf"), device=self.device, dtype=self.dtype)
                    
                    # Find minimum step size
                    for j in range(n_components):
                        if j not in active:
                            cj = components[j] @ residual
                            aj = components[j] @ u
                            
                            # Compute possible step sizes
                            plus = (C - cj) / (A - aj + 1e-10) if A > aj else float("inf")
                            minus = (C + cj) / (A + aj + 1e-10) if A > -aj else float("inf")
                            
                            # Update gamma
                            if plus < gamma:
                                gamma = plus
                            if minus < gamma:
                                gamma = minus
                    
                    # Update solution and residual
                    mu_update = torch.zeros(n_components, device=self.device, dtype=self.dtype)
                    mu_update[active] = gamma * w * signs
                    mu += mu_update
                    residual -= gamma * u
                    
                    # Check if any coefficient becomes zero
                    zero_mask = torch.abs(mu[active]) < 1e-10
                    if zero_mask.any():
                        active = active[~zero_mask]
                        signs = signs[~zero_mask]
                
                code[start_idx + i] = mu
                
        return code
    
    def _coordinate_descent(self, X: Tensor, components: Tensor, alpha: float) -> Tensor:
        """Coordinate descent for sparse coding.
        
        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
            Data to encode.
            
        components : Tensor of shape (n_components, n_features)
            Current components.
            
        alpha : float
            Sparsity controlling parameter.
            
        Returns
        -------
        code : Tensor of shape (n_samples, n_components)
            Sparse code.
        """
        n_samples, n_features = X.shape
        n_components = components.shape[0]
        
        # Precompute Gram matrix
        Gram = components @ components.T
        
        # Initialize code
        code = torch.zeros(n_samples, n_components, device=self.device, dtype=self.dtype)
        
        # Process in batches to reduce memory usage
        batch_size = min(100, n_samples)  # Adjust batch size as needed
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            
            # Solve for each sample in the batch
            for i in range(X_batch.shape[0]):
                # Initialize variables
                mu = torch.zeros(n_components, device=self.device, dtype=self.dtype)  # Current solution
                residual = X_batch[i].clone()  # Current residual
                
                # Coordinate descent loop
                for _ in range(100):  # Max inner iterations
                    max_update = 0.0
                    
                    for j in range(n_components):
                        # Compute update
                        old_mu_j = mu[j].clone()
                        numerator = components[j] @ residual + old_mu_j * Gram[j, j]
                        
                        # Soft thresholding
                        if numerator > alpha:
                            mu[j] = (numerator - alpha) / Gram[j, j]
                        elif numerator < -alpha:
                            mu[j] = (numerator + alpha) / Gram[j, j]
                        else:
                            mu[j] = 0.0
                        
                        # Update residual
                        update = mu[j] - old_mu_j
                        if abs(update) > 1e-10:
                            residual -= update * components[j]
                        
                        # Track maximum update
                        max_update = max(max_update, abs(update))
                    
                    # Check convergence
                    if max_update < 1e-4:
                        break
                
                code[start_idx + i] = mu
                
        return code


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples, n_features = 100, 10
    X = torch.randn(n_samples, n_features, dtype=torch.float64)
    
    # Create and fit SparsePCA
    spca = SparsePCA(
        n_components=5,
        alpha=0.1,
        ridge_alpha=0.01,
        max_iter=50,
        tol=1e-6,
        method="cd",
        cpu=True,  # Force CPU usage to avoid CUDA issues
        dtype=torch.float64,
        random_state=42,
    )
    
    # Fit and transform
    X_transformed = spca.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Components shape: {spca.components_.shape}")
    print(f"Number of iterations: {spca.n_iter_}")
    if spca.n_iter_ > 0:
        print(f"Final reconstruction error: {spca.error_[-1]:.6f}")
    
    # Test transform on new data
    X_new = torch.randn(10, n_features, dtype=torch.float64)
    X_new_transformed = spca.transform(X_new)
    
    print(f"New data transformed shape: {X_new_transformed.shape}")

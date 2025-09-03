import torch
from torch import Tensor

import numpy as np
from numpy import ndarray
from typing import Union, Optional, Tuple
import warnings

from models.base import BaseDimensionalityReduction


class FactorAnalysis(BaseDimensionalityReduction):
    """
    Factor Analysis (FA)
    
    A statistical method used to describe variability among observed, correlated variables
    in terms of a potentially lower number of unobserved variables called factors.
    
    Parameters:
    -----------
    n_components : int
        Number of factors to extract
    max_iter : int, default=1000
        Maximum number of iterations for the EM algorithm
    tol : float, default=1e-8
        Tolerance for the convergence of the EM algorithm
    rotation : str, default=None
        Rotation method to apply to factors ('varimax', etc.)    # ToDo : 'quartimax'
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
    components_ : Tensor of shape (n_components, n_features)
        Factor loadings matrix
    noise_variance_ : Tensor of shape (n_features,)
        Estimated noise variance for each feature
    log_likelihood_ : float
        Log-likelihood of the model
    n_iter_ : int
        Number of iterations run
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 1000,
        tol: float = 1e-8,
        rotation: Optional[str] = None,
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
        self.max_iter = max_iter
        self.tol = tol
        self.rotation = rotation
        self.noise_variance_ = None
        self.log_likelihood_ = None
        self.n_iter_ = 0
        
    def __str__(self) -> str:
        return "FactorAnalysis"
    
    def _expectation_step(self, X: Tensor, loadings: Tensor, psi: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Expectation step of the EM algorithm"""
        n_samples, n_features = X.shape
        n_factors = loadings.shape[1]
        
        # Compute the covariance matrix
        sigma = loadings @ loadings.T + torch.diag(psi)
        
        try:
            sigma_inv = torch.inverse(sigma)
        except RuntimeError:
            # Add small regularization if matrix is singular
            sigma_reg = sigma + torch.eye(n_features, device=self.device) * 1e-6
            sigma_inv = torch.inverse(sigma_reg)
        
        # Compute the factor scores using the regression method
        beta = loadings.T @ sigma_inv  # Regression coefficients
        scores = X @ beta.T  # Factor scores
        
        # Compute the expected sufficient statistics
        xxT = X.T @ X
        xsT = X.T @ scores
        ssT = scores.T @ scores + n_samples * (torch.eye(n_factors, device=self.device) - beta @ loadings)
        
        return xsT, ssT, xxT
    
    def _maximization_step(self, xsT: Tensor, ssT: Tensor, xxT: Tensor, n_samples: int) -> Tuple[Tensor, Tensor]:
        """Maximization step of the EM algorithm"""
        n_features, n_factors = xsT.shape
        
        # Update factor loadings
        try:
            loadings = xsT @ torch.inverse(ssT)
        except RuntimeError:
            # Add small regularization if matrix is singular
            ssT_reg = ssT + torch.eye(n_factors, device=self.device) * 1e-6
            loadings = xsT @ torch.inverse(ssT_reg)
        
        # Update noise variances
        psi = torch.diag(xxT - loadings @ xsT.T) / n_samples
        psi = torch.clamp(psi, min=1e-6)  # Ensure positive variance
        
        return loadings, psi
    
    def _varimax_rotation(self, loadings: Tensor, gamma: float = 1.0, max_iter: int = 100, tol: float = 1e-6) -> Tensor:
        """Apply varimax rotation to factor loadings"""
        n_features, n_factors = loadings.shape
        
        # Normalize the loadings matrix
        norm_factor = torch.sqrt(torch.sum(loadings**2, dim=1, keepdim=True))
        norm_factor = torch.clamp(norm_factor, min=1e-8)  # Avoid division by zero
        loadings_norm = loadings / norm_factor
        
        # Initialize rotation matrix
        R = torch.eye(n_factors, device=self.device, dtype=self.dtype)
        
        for i in range(max_iter):
            # Compute the rotated loadings
            loadings_rot = loadings_norm @ R
            
            # Compute the objective function
            lambda_sq = loadings_rot**2
            obj_prev = torch.sum(torch.var(lambda_sq, dim=0))
            
            # Update the rotation matrix
            for j in range(n_factors):
                for k in range(j + 1, n_factors):
                    # Compute the submatrix for factors j and k
                    u = loadings_norm[:, j]
                    v = loadings_norm[:, k]
                    
                    # Compute the rotation angle
                    num = 2 * torch.sum(u * v * (u**2 - v**2) - gamma * u * v)
                    den = torch.sum((u**2 - v**2)**2 - 4 * u**2 * v**2 - gamma * (u**2 - v**2))
                    theta = 0.25 * torch.atan2(num, den)
                    
                    # Create the rotation matrix for factors j and k
                    R_jk = torch.eye(n_factors, device=self.device, dtype=self.dtype)
                    R_jk[j, j] = torch.cos(theta)
                    R_jk[j, k] = -torch.sin(theta)
                    R_jk[k, j] = torch.sin(theta)
                    R_jk[k, k] = torch.cos(theta)
                    
                    # Update the rotation matrix
                    R = R @ R_jk
            
            # Check convergence
            loadings_rot = loadings_norm @ R
            lambda_sq = loadings_rot**2
            obj = torch.sum(torch.var(lambda_sq, dim=0))
            
            if torch.abs(obj - obj_prev) < tol:
                break
        
        # Apply the rotation to the original loadings
        return loadings @ R
    
    def fit(self, X: Union[ndarray, Tensor]) -> "FactorAnalysis":
        """Fit the Factor Analysis model with X using the EM algorithm"""
        X = self.check_input(X)
        n_samples, n_features = X.shape

        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        
        # Initialize parameters
        loadings = torch.randn(n_features, self.n_components, device=self.device, dtype=self.dtype) * 0.1
        psi = torch.ones(n_features, device=self.device, dtype=self.dtype)
        
        # EM algorithm
        prev_log_likelihood = -float('inf')
        self.log_likelihood_ = -float('inf')
        
        for i in range(self.max_iter):
            try:
                # E-step
                xsT, ssT, xxT = self._expectation_step(X_centered, loadings, psi)
                
                # M-step
                loadings, psi = self._maximization_step(xsT, ssT, xxT, n_samples)
                
                # Compute log-likelihood
                sigma = loadings @ loadings.T + torch.diag(psi)
                
                try:
                    sigma_inv = torch.inverse(sigma)
                    log_det_sigma = torch.logdet(sigma)
                    
                    # Log-likelihood: -n/2 * [log|sigma| + tr(sigma_inv * cov) + p*log(2π)]
                    cov = X_centered.T @ X_centered / n_samples
                    log_likelihood = -0.5 * n_samples * (log_det_sigma + torch.trace(sigma_inv @ cov) + 
                                                        n_features * torch.log(torch.tensor(2 * np.pi, device=self.device)))
                    
                    # Check convergence
                    if torch.abs(log_likelihood - prev_log_likelihood) < self.tol:
                        self.log_likelihood_ = log_likelihood.item()
                        break
                        
                    prev_log_likelihood = log_likelihood
                    self.log_likelihood_ = log_likelihood.item()
                    
                except RuntimeError:
                    # Skip log-likelihood calculation if matrix inversion fails
                    warnings.warn(f"Log-likelihood calculation failed at iteration {i}")
                    
            except RuntimeError as e:
                warnings.warn(f"EM algorithm failed at iteration {i}: {e}")
                break
                
            self.n_iter_ = i + 1
        
        # Apply rotation if specified
        if self.rotation == "varimax":
            loadings = self._varimax_rotation(loadings)
        
        self.components_ = loadings.T  # Store as (n_components, n_features)
        self.noise_variance_ = psi
        
        return self
    
    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Transform X to the factor space"""
        if self.components_ is None:
            raise RuntimeError("FactorAnalysis must be fitted before transforming data")
            
        X = self.check_input(X)
        X_centered = X - self.mean_
        
        # Compute factor scores using regression method
        loadings = self.components_.T  # (n_features, n_components)
        
        # Compute the covariance matrix
        sigma = loadings @ loadings.T + torch.diag(self.noise_variance_)
        
        try:
            sigma_inv = torch.inverse(sigma)
        except RuntimeError:
            # Add small regularization if matrix is singular
            sigma_reg = sigma + torch.eye(sigma.shape[0], device=self.device) * 1e-6
            sigma_inv = torch.inverse(sigma_reg)
        
        # Compute regression coefficients
        beta = loadings.T @ sigma_inv  # (n_components, n_features)
        
        # Compute factor scores
        scores = X_centered @ beta.T  # (n_samples, n_components)
        
        return scores
    
    def get_covariance(self) -> Tensor:
        """Compute the covariance matrix of the fitted model"""
        if self.components_ is None:
            raise RuntimeError("FactorAnalysis must be fitted before getting covariance")
            
        loadings = self.components_.T  # (n_features, n_components)
        return loadings @ loadings.T + torch.diag(self.noise_variance_)


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples, n_features, n_factors = 1000, 100, 5

    true_loadings = torch.randn(n_features, n_factors, dtype=torch.float64)
    true_scores = torch.randn(n_samples, n_factors, dtype=torch.float64)
    noise = torch.randn(n_samples, n_features, dtype=torch.float64) * 0.1
    X = true_scores @ true_loadings.T + noise
    
    # Test FactorAnalysis
    fa = FactorAnalysis(
        n_components=n_factors, 
        max_iter=100, 
        rotation="varimax",
        # cpu=True,
    )
    X_transformed = fa.fit_transform(X)
    
    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)
    print("Log-likelihood:", fa.log_likelihood_)
    print("Number of iterations:", fa.n_iter_)
    print("Noise variance shape:", fa.noise_variance_.shape)
    print("Components shape:", fa.components_.shape)

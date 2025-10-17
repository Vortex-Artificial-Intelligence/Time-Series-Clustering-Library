import torch
from torch import Tensor

from numpy import ndarray
from typing import Union, Optional, Tuple

from models.base import BaseDimensionalityReduction


class TSNE(BaseDimensionalityReduction):
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE)

    A nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data
    into a space of two or three dimensions, which can then be visualized in a scatter plot.

    Parameters:
    -----------
    n_components : int
        Dimension of the embedded space (usually 2 or 3)
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in the embedded space
    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]
    n_iter : int, default=1000
        Maximum number of iterations for the optimization
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the optimization
    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will be stopped
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances in a feature array
        ('euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine')
    p : int, default=3
        Distance parameter for minkowski, (1=Manhattan, 2=Euclidean, ∞=Chebyshev)
    init : str or Tensor, default='random'
        Initialization of embedding. Possible options are 'random', 'pca', or a tensor
    verbose : int, default=0
        Verbosity level
    random_state : int, default=42
        Random seed for reproducibility
    method : str, default='exact'
        Gradient calculation method ('exact' or 'barnes_hut')
    angle : float, default=0.5
        Only used if method='barnes_hut'. Controls trade-off between speed and accuracy
    cpu : bool, default=False
        If True, use CPU only
    device : int, default=0
        CUDA device index
    dtype : torch.dtype, default=torch.float64
        Data type for computations

    Attributes:
    -----------
    embedding_ : Tensor of shape (n_samples, n_components)
        Stores the embedding vectors
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization
    n_iter_ : int
        Number of iterations run
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        n_iter_without_progress: int = 300,
        min_grad_norm: float = 1e-7,
        metric: str = "euclidean",
        p: int = 3,
        init: Union[str, Tensor] = "random",
        verbose: int = 0,
        random_state: int = 42,
        method: str = "exact",
        angle: float = 0.5,
        cpu: Optional[bool] = False,
        device: Optional[int] = 0,
        dtype: Optional[torch.dtype] = torch.float64,
    ) -> None:
        super().__init__(
            n_components=n_components,
            cpu=cpu,
            device=device,
            dtype=dtype,
            random_state=random_state,
        )
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.p = (p,)
        self.init = init
        self.verbose = verbose
        self.method = method
        self.angle = angle
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = 0

        if self.method not in ["exact", "barnes_hut"]:
            raise ValueError(
                f"method must be 'exact' or 'barnes_hut', got {self.method}"
            )

        if self.metric not in [
            "euclidean",
            "manhattan",
            "chebyshev",
            "minkowski",
            "cosine",
        ]:
            raise ValueError(
                f"metric must be 'euclidean', 'manhattan', 'chebyshev', 'minkowski', or 'cosine', got {self.metric}"
            )

    def __str__(self) -> str:
        return "TSNE"

    def _compute_distance(self, X: Tensor, Y: Optional[Tensor] = None) -> Tensor:
        """Compute pairwise distances according to the specified metric"""
        if Y is None:
            Y = X

        if self.metric == "euclidean":
            return torch.cdist(X, Y, p=2)
        elif self.metric == "manhattan":
            return torch.cdist(X, Y, p=1)
        elif self.metric == "chebyshev":
            return torch.cdist(X, Y, p=float("inf"))
        elif self.metric == "minkowski":
            return torch.cdist(X, Y, p=self.p)
        elif self.metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            X_norm = X / torch.norm(X, dim=1, keepdim=True)
            if Y is X:
                Y_norm = X_norm
            else:
                Y_norm = Y / torch.norm(Y, dim=1, keepdim=True)
            return 1 - torch.mm(X_norm, Y_norm.T)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _hbeta(self, D: Tensor, beta: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Compute H and P for a given beta value"""
        P = torch.exp(-D * beta)
        sumP = torch.sum(P)
        H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
        P = P / sumP
        return H, P

    def _x2p(self, X: Tensor, tol: float = 1e-5, perplexity: float = 30.0) -> Tensor:
        """Compute pairwise affinities for t-SNE"""
        n = X.shape[0]
        P = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        beta = torch.ones(n, device=self.device, dtype=self.dtype)
        logU = torch.log(torch.tensor(perplexity, device=self.device, dtype=self.dtype))

        # Compute pairwise distances according to the specified metric
        D = self._compute_distance(X)

        # Loop over all datapoints
        for i in range(n):
            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -float("inf")
            betamax = float("inf")
            Di = D[i, torch.arange(n) != i]

            H, thisP = self._hbeta(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while torch.abs(Hdiff) > tol and tries < 50:
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].clone()
                    if betamax == float("inf"):
                        beta[i] = beta[i] * 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i].clone()
                    if betamin == -float("inf"):
                        beta[i] = beta[i] / 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2

                # Recompute the values
                H, thisP = self._hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, torch.arange(n) != i] = thisP

        return P

    def _pca_init(self, X: Tensor) -> Tensor:
        """Initialize embedding using PCA"""
        n_samples, n_features = X.shape

        # Center the data
        mean = torch.mean(X, dim=0)
        X_centered = X - mean

        # Compute covariance matrix
        cov = X_centered.T @ X_centered / n_samples

        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Project to n_components
        embedding = X_centered @ eigenvectors[:, : self.n_components]

        return embedding

    def _exact_tsne_grad(self, P: Tensor, Y: Tensor) -> Tuple[Tensor, float]:
        """Compute exact gradient of t-SNE cost function"""
        n = Y.shape[0]

        # Compute Q (student-t distribution)
        sum_Y = torch.sum(Y**2, dim=1)
        D = sum_Y.unsqueeze(1) + sum_Y.unsqueeze(0) - 2 * Y @ Y.T
        Q = 1 / (1 + D)
        Q.fill_diagonal_(0)
        Q = Q / torch.sum(Q)

        # Compute gradient
        PQ = P - Q
        grad = 4 * (PQ * Q) @ Y - 4 * torch.diag(torch.sum(PQ * Q, dim=1)) @ Y

        # Compute KL divergence
        kl_divergence = torch.sum(P * torch.log((P + 1e-12) / (Q + 1e-12)))

        return grad, kl_divergence

    def _barnes_hut_tsne_grad(self, P: Tensor, Y: Tensor) -> Tuple[Tensor, float]:
        """Compute Barnes-Hut approximation of t-SNE gradient"""
        n = Y.shape[0]

        # Compute Q (student-t distribution)
        sum_Y = torch.sum(Y**2, dim=1)
        D = sum_Y.unsqueeze(1) + sum_Y.unsqueeze(0) - 2 * Y @ Y.T
        Q = 1 / (1 + D)
        Q.fill_diagonal_(0)
        Q = Q / torch.sum(Q)

        # Compute exact forces for nearest neighbors
        PQ = P - Q
        grad = 4 * (PQ * Q) @ Y - 4 * torch.diag(torch.sum(PQ * Q, dim=1)) @ Y

        # For Barnes-Hut, we approximate the long-range forces using a quadtree
        # Since implementing a full quadtree in PyTorch is complex, we use a simpler approach:
        # We divide the embedding space into cells and approximate forces from distant cells

        # Simple grid-based approximation
        if self.n_components == 2:
            # For 2D embeddings, we can use a grid-based approximation
            min_coords = torch.min(Y, dim=0)[0]
            max_coords = torch.max(Y, dim=0)[0]

            # Create a grid
            grid_size = int(torch.sqrt(torch.tensor(n, dtype=self.dtype)).item())
            grid_cells = torch.linspace(
                min_coords[0], max_coords[0], grid_size + 1, device=self.device
            )
            grid_cells_y = torch.linspace(
                min_coords[1], max_coords[1], grid_size + 1, device=self.device
            )

            # Compute cell centers and masses
            cell_centers = torch.zeros(
                (grid_size, grid_size, 2), device=self.device, dtype=self.dtype
            )
            cell_masses = torch.zeros(
                (grid_size, grid_size), device=self.device, dtype=self.dtype
            )

            for i in range(grid_size):
                for j in range(grid_size):
                    x_min, x_max = grid_cells[i], grid_cells[i + 1]
                    y_min, y_max = grid_cells_y[j], grid_cells_y[j + 1]

                    # Find points in this cell
                    in_cell = (
                        (Y[:, 0] >= x_min)
                        & (Y[:, 0] < x_max)
                        & (Y[:, 1] >= y_min)
                        & (Y[:, 1] < y_max)
                    )

                    if torch.any(in_cell):
                        cell_centers[i, j] = torch.mean(Y[in_cell], dim=0)
                        cell_masses[i, j] = torch.sum(Q[:, in_cell], dim=1).sum()

            # Approximate forces from distant cells
            for i in range(n):
                point = Y[i]
                exact_force = grad[i].clone()

                # For each cell, check if we can approximate
                for cell_i in range(grid_size):
                    for cell_j in range(grid_size):
                        if cell_masses[cell_i, cell_j] == 0:
                            continue

                        cell_center = cell_centers[cell_i, cell_j]
                        distance = torch.norm(point - cell_center)

                        # Cell size (approximate)
                        cell_size = max(
                            grid_cells[1] - grid_cells[0],
                            grid_cells_y[1] - grid_cells_y[0],
                        )

                        # Check if we can approximate this cell
                        if cell_size / distance < self.angle:
                            # Approximate force from this cell
                            q_cell = cell_masses[cell_i, cell_j]
                            force_dir = point - cell_center
                            force_magnitude = 4 * q_cell / (1 + distance**2)
                            approx_force = force_magnitude * force_dir / distance

                            # Replace exact forces with approximation
                            # This is a simplified version - in practice we'd need to be more careful
                            grad[i] = grad[i] + approx_force - exact_force

        # Compute KL divergence
        kl_divergence = torch.sum(P * torch.log((P + 1e-12) / (Q + 1e-12)))

        return grad, kl_divergence

    def _tsne_grad(self, P: Tensor, Y: Tensor) -> Tuple[Tensor, float]:
        """Compute gradient of t-SNE cost function using the specified method"""
        if self.method == "exact":
            return self._exact_tsne_grad(P, Y)
        elif self.method == "barnes_hut":
            return self._barnes_hut_tsne_grad(P, Y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X: Union[ndarray, Tensor]) -> "TSNE":
        """Fit the t-SNE model with X"""
        X = self.check_input(X)
        n_samples, n_features = X.shape

        # Check if Barnes-Hut is applicable
        if self.method == "barnes_hut" and self.n_components not in [2, 3]:
            if self.verbose:
                print(
                    "Barnes-Hut is only applicable to 2D or 3D embeddings. Switching to exact method."
                )
            self.method = "exact"

        # Compute pairwise affinities
        if self.verbose:
            print("Computing pairwise affinities...")
        P = self._x2p(X, perplexity=self.perplexity)
        P = (P + P.T) / (2 * n_samples)  # Symmetrize

        # Initialize the embedding
        if isinstance(self.init, Tensor):
            Y = self.init.clone()
        elif self.init == "pca":
            Y = self._pca_init(X)
        else:  # random
            Y = torch.randn(
                n_samples, self.n_components, device=self.device, dtype=self.dtype
            )

        # Early exaggeration
        P *= self.early_exaggeration

        # Initialize optimization variables
        gains = torch.ones_like(Y)
        iY = torch.zeros_like(Y)

        # Optimization loop
        best_kl = float("inf")
        best_Y = Y.clone()
        no_improvement = 0

        if self.verbose:
            print("Optimizing t-SNE...")

        for i in range(self.n_iter):
            # Compute gradient and KL divergence
            grad, kl_divergence = self._tsne_grad(P, Y)

            # Check for improvement
            if kl_divergence < best_kl:
                best_kl = kl_divergence
                best_Y = Y.clone()
                no_improvement = 0
            else:
                no_improvement += 1

            # Stop if no improvement for too long
            if no_improvement >= self.n_iter_without_progress:
                if self.verbose:
                    print(f"Early stopping at iteration {i}")
                break

            # Update gains
            gains = (gains + 0.2) * ((grad > 0) != (iY > 0)).float() + (gains * 0.8) * (
                (grad > 0) == (iY > 0)
            ).float()
            gains.clamp_min_(0.01)

            # Update embedding
            iY = self.learning_rate * gains * grad
            Y += iY

            # Center the embedding
            Y = Y - torch.mean(Y, dim=0)

            # Stop if gradient norm is too small
            grad_norm = torch.norm(grad)
            if grad_norm < self.min_grad_norm:
                if self.verbose:
                    print(f"Gradient norm below threshold at iteration {i}")
                break

            # Reduce early exaggeration after initial period
            if i == 100:
                P /= self.early_exaggeration

            # Print progress
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}, KL divergence: {kl_divergence:.4f}")

        self.embedding_ = best_Y
        self.kl_divergence_ = best_kl
        self.n_iter_ = i + 1

        return self

    def transform(self, X: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
        """Transform X to the embedded space"""
        if self.embedding_ is None:
            raise RuntimeError("TSNE must be fitted before transforming data")

        # For t-SNE, we can't transform new points directly
        # We need to recompute the embedding including the new points
        # This is a limitation of t-SNE
        raise NotImplementedError(
            "TSNE does not support transforming new data. Use fit_transform instead."
        )

    def fit_transform(self, X: Union[ndarray, Tensor], *args) -> Union[ndarray, Tensor]:
        """Fit the model with X and apply the dimensionality reduction on X"""
        self.fit(X)
        return self.embedding_


# Test
if __name__ == "__main__":
    torch.manual_seed(42)
    n_samples, n_features = 1000, 50
    X = torch.randn(n_samples, n_features, dtype=torch.float64)

    # Test t-SNE with different methods and metrics
    methods = ["exact", "barnes_hut"]
    metrics = ["euclidean", "manhattan", "cosine"]

    for method in methods:
        for metric in metrics:
            print(f"\nTesting method={method}, metric={metric}")
            tsne = TSNE(
                n_components=2,
                perplexity=30,
                n_iter=500,
                verbose=1,
                method=method,
                metric=metric,
            )
            X_transformed = tsne.fit_transform(X)
            
            print(f"Original shape: {X.shape}")
            print(f"Transformed shape: {X_transformed.shape}")
            print(f"KL divergence: {tsne.kl_divergence_}")
            print(f"Number of iterations: {tsne.n_iter_}")
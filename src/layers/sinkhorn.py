"""
Sinkhorn-Knopp algorithm for projecting matrices onto the Birkhoff polytope.

The Birkhoff polytope B_n is the set of n×n doubly stochastic matrices:
    B_n = {H ∈ R^{n×n} : H1 = 1, H^T 1 = 1, H ≥ 0}

where 1 is the all-ones vector. Doubly stochastic matrices have:
    - All rows sum to 1
    - All columns sum to 1  
    - All entries non-negative

THEORETICAL IMPORTANCE:
----------------------
The Sinkhorn projection is CRITICAL for mHC-GNN's theoretical guarantees:

1. Theorem 1 (Over-smoothing): Requires H^res ∈ B_n to ensure the product
   of mixing matrices across layers remains doubly stochastic, preserving
   feature diversity and achieving the slower convergence rate (1-γ)^{L/n}

2. Theorem 2 (Expressiveness): Doubly stochastic mixing preserves distinct
   stream information, enabling detection of structural patterns beyond 1-WL

ALGORITHM (Paper Eq. 9-10):
--------------------------
Starting from M^(0) = exp(H/τ):
    M^(t) ← diag(M^(t-1) · 1)^{-1} · M^(t-1}     [row normalization]
    M^(t) ← M^(t) · diag(M^(t)^T · 1)^{-1}       [column normalization]

After T iterations, M^(T) approximates a doubly stochastic matrix.
"""

import torch
import torch.nn as nn


def sinkhorn_knopp(H_raw, tau=0.1, num_iters=20, eps=1e-8):
    """
    Project matrix H_raw onto the Birkhoff polytope of doubly stochastic matrices.
    
    This implements the Sinkhorn-Knopp algorithm (Paper Eq. 9-10, Eq. 21).
    The algorithm alternates between row and column normalization, which
    provably converges to the nearest doubly stochastic matrix.

    Args:
        H_raw: Input matrix of shape (..., n, n) - can be batched
               This is the raw (pre-projection) mixing matrix Ĥ^res
        tau: Temperature parameter for softmax (default 0.1)
             Lower τ → sharper (more permutation-like) matrices
             Higher τ → softer (more uniform) matrices
        num_iters: Number of Sinkhorn iterations (default 20)
                   More iterations → better approximation to doubly stochastic
                   10-20 iterations typically sufficient for 1e-3 error
        eps: Small constant for numerical stability

    Returns:
        Doubly stochastic matrix of shape (..., n, n)
        Guaranteed: row sums ≈ 1, column sums ≈ 1, all entries ≥ 0
    """
    # ============================================================
    # Step 1: Apply temperature-scaled exp to ensure positivity
    # Paper: M^(0) = exp(H/τ)
    # Temperature τ controls the "sharpness" of the resulting matrix
    # ============================================================
    M = torch.exp(H_raw / tau)

    # Numerical stability: clamp to avoid division by zero
    M = torch.clamp(M, min=eps)

    # ============================================================
    # Step 2: Alternating row/column normalization (Paper Eq. 9-10)
    # This is the core Sinkhorn-Knopp iteration
    # Convergence is guaranteed for positive matrices
    # ============================================================
    for _ in range(num_iters):
        # Row normalization: make each row sum to 1
        # Paper Eq. 9: M ← diag(M·1)^{-1} · M
        row_sums = M.sum(dim=-1, keepdim=True)
        M = M / (row_sums + eps)

        # Column normalization: make each column sum to 1
        # Paper Eq. 10: M ← M · diag(M^T·1)^{-1}
        col_sums = M.sum(dim=-2, keepdim=True)
        M = M / (col_sums + eps)

    return M


class SinkhornProjection(nn.Module):
    """
    Learnable Sinkhorn projection module.
    """
    def __init__(self, tau=0.1, num_iters=10):
        super().__init__()
        self.tau = tau
        self.num_iters = num_iters

    def forward(self, H_raw):
        """
        Args:
            H_raw: Raw matrix of shape (batch, n, n) or (n, n)

        Returns:
            Doubly stochastic matrix
        """
        return sinkhorn_knopp(H_raw, self.tau, self.num_iters)

    def extra_repr(self):
        return f'tau={self.tau}, num_iters={self.num_iters}'


def verify_doubly_stochastic(H, tol=1e-3):
    """
    Verify that H is approximately doubly stochastic.

    Args:
        H: Matrix of shape (..., n, n)
        tol: Tolerance for row/column sum deviation from 1

    Returns:
        Dictionary with verification results
    """
    row_sums = H.sum(dim=-1)
    col_sums = H.sum(dim=-2)

    row_error = (row_sums - 1.0).abs().max().item()
    col_error = (col_sums - 1.0).abs().max().item()

    is_doubly_stochastic = (row_error < tol) and (col_error < tol)

    return {
        'is_doubly_stochastic': is_doubly_stochastic,
        'row_error': row_error,
        'col_error': col_error,
        'row_sum_mean': row_sums.mean().item(),
        'col_sum_mean': col_sums.mean().item(),
    }


if __name__ == "__main__":
    # Test Sinkhorn projection
    print("Testing Sinkhorn-Knopp projection...")

    # Random matrix
    n = 4
    H_raw = torch.randn(2, n, n)  # batch of 2

    # Apply Sinkhorn
    H_ds = sinkhorn_knopp(H_raw, tau=0.1, num_iters=10)

    # Verify
    results = verify_doubly_stochastic(H_ds)
    print(f"Doubly stochastic: {results['is_doubly_stochastic']}")
    print(f"Row error: {results['row_error']:.6f}")
    print(f"Col error: {results['col_error']:.6f}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    H_raw = torch.randn(n, n, requires_grad=True)
    H_ds = sinkhorn_knopp(H_raw)
    loss = H_ds.sum()
    loss.backward()
    print(f"Gradient exists: {H_raw.grad is not None}")
    print(f"Gradient norm: {H_raw.grad.norm().item():.6f}")

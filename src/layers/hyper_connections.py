"""
Manifold-Constrained Hyper-Connections layer for GNNs.

This implements the mHC-GNN layer from the paper. The key theoretical guarantee
is that H^res is projected onto the Birkhoff polytope (doubly stochastic matrices)
via the Sinkhorn-Knopp algorithm, which ensures:
  1. Slower over-smoothing rate: (1-γ)^{L/n} vs (1-γ)^L for standard GNNs
  2. Expressiveness beyond 1-WL through multi-stream representations

IMPLEMENTATION NOTES vs PAPER:
------------------------------
The implementation makes some simplifications that do NOT affect the core 
theoretical guarantees (proofs only require H^res ∈ Birkhoff polytope):

1. Activation: Uses tanh instead of sigmoid (σ) in paper Eq. 18-19
   - Both are bounded smooth functions, proofs are activation-agnostic
   - TO MATCH PAPER: Replace torch.tanh() with torch.sigmoid()

2. H^post scaling: Missing factor of 2 from paper Eq. 19
   - Paper: H^post = 2σ(...)  
   - Code: H^post = tanh(...)
   - TO MATCH PAPER: Multiply H_post by 2 after computation

3. H^res dynamic: Uses rank-1 broadcast instead of full (n×n) dynamic matrix
   - Paper Eq. 20: Θ_res (n×d) @ x̃^T (d×n) → (n×n) per node
   - Code: x_pooled (N×d) @ Θ_res^T (d×n) → (N×n), broadcast to (N,n,n)
   - This is more parameter-efficient; Sinkhorn still guarantees doubly stochastic
   - TO MATCH PAPER: Use outer product or per-stream projections

4. Normalization: Uses L2 norm instead of RMSNorm
   - Paper Eq. 17: x̃ = RMSNorm(x)
   - Code: F.normalize(x, p=2, dim=-1)
   - Both serve similar stability purposes
   - TO MATCH PAPER: Implement RMSNorm = x / sqrt(mean(x²) + ε)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .sinkhorn import sinkhorn_knopp
except ImportError:
    from sinkhorn import sinkhorn_knopp


class HyperConnection(nn.Module):
    """
    Single hyper-connection layer implementing mHC for GNNs.

    Maintains n parallel streams and uses doubly stochastic matrices for stream mixing.
    
    The layer update rule (paper Eq. 15):
        x^{l+1} = H^res · x^{l} + (H^post)^T · F_GNN(H^pre · x^{l}, neighbors)
    
    where:
        - H^res ∈ R^{n×n}: Stream mixing matrix (MUST be doubly stochastic for theory)
        - H^pre ∈ R^{1×n}: Stream aggregation weights for GNN input
        - H^post ∈ R^{1×n}: Stream expansion weights for GNN output
    """
    def __init__(
        self,
        n_streams,
        hidden_dim,
        sinkhorn_tau=0.1,
        sinkhorn_iters=10,
        use_dynamic=True,
        use_static=True,
        init_alpha=0.01,
    ):
        super().__init__()
        self.n_streams = n_streams
        self.hidden_dim = hidden_dim
        self.sinkhorn_tau = sinkhorn_tau
        self.sinkhorn_iters = sinkhorn_iters
        self.use_dynamic = use_dynamic
        self.use_static = use_static

        # ============================================================
        # Learnable parameters for H^pre (1 x n) - stream aggregation
        # Paper Eq. 18: H^pre = σ(α · (θ · x̃^T) + b)
        # ============================================================
        if use_dynamic:
            # θ_pre ∈ R^{1×d}: projects pooled features to scalar
            self.theta_pre = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
        if use_static:
            # b_pre ∈ R^{1×n}: static bias, init uniform for equal stream weighting
            self.b_pre = nn.Parameter(torch.ones(1, n_streams) / n_streams)

        # ============================================================
        # Learnable parameters for H^post (1 x n) - stream expansion  
        # Paper Eq. 19: H^post = 2σ(α · (θ · x̃^T) + b)
        # NOTE: Missing factor of 2 in implementation
        # ============================================================
        if use_dynamic:
            self.theta_post = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
        if use_static:
            self.b_post = nn.Parameter(torch.ones(1, n_streams) / n_streams)

        # ============================================================
        # Learnable parameters for H^res (n x n) - stream mixing
        # Paper Eq. 20-21: Ĥ^res = α · (Θ · x̃^T) + B, then Sinkhorn
        # CRITICAL: Sinkhorn projection ensures H^res ∈ Birkhoff polytope
        # ============================================================
        if use_dynamic:
            # Θ_res ∈ R^{n×d}: projects to n-dim, then broadcast to (n×n)
            # TO MATCH PAPER: Would need Θ_res ∈ R^{n×n×d} for full dynamic
            self.Theta_res = nn.Parameter(torch.randn(n_streams, hidden_dim) * 0.01)
        if use_static:
            # B_res ∈ R^{n×n}: static bias, init near identity for residual-like behavior
            self.B_res = nn.Parameter(torch.eye(n_streams) + torch.randn(n_streams, n_streams) * 0.01)

        # ============================================================
        # Gating scalars α (initialized small for smooth training start)
        # These control the dynamic component magnitude
        # ============================================================
        self.alpha_pre = nn.Parameter(torch.tensor(init_alpha))
        self.alpha_post = nn.Parameter(torch.tensor(init_alpha))
        self.alpha_res = nn.Parameter(torch.tensor(init_alpha))

    def compute_mappings(self, x):
        """
        Compute H^pre, H^post, H^res for input x.
        
        Paper Equations 17-21:
            x̃ = RMSNorm(x)                                    [Eq. 17]
            H^pre = σ(α_pre · (θ_pre · x̃^T) + b_pre)         [Eq. 18]
            H^post = 2σ(α_post · (θ_post · x̃^T) + b_post)    [Eq. 19]
            Ĥ^res = α_res · (Θ_res · x̃^T) + B_res           [Eq. 20]
            H^res = Sinkhorn(Ĥ^res, T)                        [Eq. 21]

        Args:
            x: Multi-stream node features of shape (num_nodes, n_streams, hidden_dim)

        Returns:
            H_pre: (num_nodes, 1, n_streams)
            H_post: (num_nodes, 1, n_streams)
            H_res: (num_nodes, n_streams, n_streams) - GUARANTEED doubly stochastic
        """
        num_nodes = x.size(0)

        # ============================================================
        # Step 1: Normalize input (Paper Eq. 17)
        # Implementation uses L2 norm; paper uses RMSNorm
        # TO MATCH PAPER: x_norm = x / sqrt((x**2).mean(dim=-1, keepdim=True) + 1e-8)
        # ============================================================
        x_norm = F.normalize(x, p=2, dim=-1)  # (num_nodes, n_streams, hidden_dim)

        # Aggregate across streams for dynamic component computation
        # This pools the n streams into a single representation per node
        x_pooled = x_norm.mean(dim=1)  # (num_nodes, hidden_dim)

        # ============================================================
        # Step 2: Compute H^pre (Paper Eq. 18)
        # Implementation uses tanh; paper uses sigmoid
        # TO MATCH PAPER: Replace tanh with torch.sigmoid
        # ============================================================
        H_pre = torch.zeros(num_nodes, 1, self.n_streams, device=x.device)
        if self.use_dynamic and hasattr(self, 'theta_pre'):
            # Dynamic: input-dependent component
            dynamic_pre = torch.tanh(x_pooled @ self.theta_pre.t())  # (num_nodes, 1)
            H_pre += self.alpha_pre * dynamic_pre.unsqueeze(-1)  # broadcast to (N, 1, n)
        if self.use_static and hasattr(self, 'b_pre'):
            # Static: learned bias independent of input
            H_pre += self.b_pre.unsqueeze(0)  # (1, 1, n) -> broadcast

        # ============================================================
        # Step 3: Compute H^post (Paper Eq. 19)
        # Implementation uses tanh without ×2; paper uses 2×sigmoid
        # TO MATCH PAPER: Use torch.sigmoid and multiply by 2
        # ============================================================
        H_post = torch.zeros(num_nodes, 1, self.n_streams, device=x.device)
        if self.use_dynamic and hasattr(self, 'theta_post'):
            dynamic_post = torch.tanh(x_pooled @ self.theta_post.t())
            H_post += self.alpha_post * dynamic_post.unsqueeze(-1)
        if self.use_static and hasattr(self, 'b_post'):
            H_post += self.b_post.unsqueeze(0)
        # TO MATCH PAPER: H_post = 2 * torch.sigmoid(H_post)

        # ============================================================
        # Step 4: Compute H^res (Paper Eq. 20-21)
        # THIS IS THE CRITICAL COMPONENT FOR THEORETICAL GUARANTEES
        # 
        # Implementation uses rank-1 dynamic (efficient but simplified):
        #   dynamic_res shape: (N, n) -> broadcast to (N, n, n)
        # 
        # Paper uses full-rank dynamic:
        #   Θ_res (n×d) @ x̃^T (d×n) -> (n×n) per node
        #
        # TO MATCH PAPER: Use per-node outer product or (n×n×d) params
        # ============================================================
        H_res_raw = torch.zeros(num_nodes, self.n_streams, self.n_streams, device=x.device)
        if self.use_dynamic and hasattr(self, 'Theta_res'):
            # Current: (N, d) @ (d, n) = (N, n), then broadcast to (N, n, n)
            # This creates rank-1 dynamic perturbations (same value along columns)
            dynamic_res = torch.tanh(x_pooled @ self.Theta_res.t())  # (N, n)
            H_res_raw += self.alpha_res * dynamic_res.unsqueeze(-1)  # (N, n, 1) -> (N, n, n)
            
            # TO MATCH PAPER (full-rank dynamic):
            # Option A: Outer product approach
            #   dynamic_res_full = torch.bmm(
            #       dynamic_res.unsqueeze(-1),  # (N, n, 1)
            #       dynamic_res.unsqueeze(1)    # (N, 1, n)
            #   )  # -> (N, n, n)
            # Option B: Separate projection per stream (expensive)
            
        if self.use_static and hasattr(self, 'B_res'):
            # Static bias: same for all nodes
            H_res_raw += self.B_res.unsqueeze(0)  # (1, n, n) -> broadcast to (N, n, n)

        # ============================================================
        # Step 5: Sinkhorn projection (Paper Eq. 21)
        # CRITICAL: This ensures H_res ∈ Birkhoff polytope (doubly stochastic)
        # This is what enables the theoretical guarantees in Theorem 1 & 2
        # ============================================================
        H_res = sinkhorn_knopp(H_res_raw, tau=self.sinkhorn_tau, num_iters=self.sinkhorn_iters)

        return H_pre, H_post, H_res

    def forward(self, x_streams, message_output):
        """
        Apply hyper-connection transformation.
        
        Paper Eq. 15:
            x^{l+1} = H^res · x^{l} + (H^post)^T · F_GNN(...)
        
        Note: H^pre is returned but aggregation happens in mhc_gnn.py
              Currently uses mean aggregation instead of H^pre @ x

        Args:
            x_streams: Input multi-stream features (num_nodes, n_streams, hidden_dim)
            message_output: Output from GNN message passing (num_nodes, hidden_dim)

        Returns:
            x_streams_new: Updated multi-stream features (num_nodes, n_streams, hidden_dim)
            H_pre: Pre-aggregation weights (for potential use in caller)
        """
        num_nodes = x_streams.size(0)

        # Compute all mapping matrices
        H_pre, H_post, H_res = self.compute_mappings(x_streams)

        # ============================================================
        # Residual path: H^res @ x_streams
        # H_res is GUARANTEED doubly stochastic by Sinkhorn projection
        # This preserves feature diversity across streams (key for Theorem 1)
        # ============================================================
        # H_res: (num_nodes, n_streams, n_streams)
        # x_streams: (num_nodes, n_streams, hidden_dim)
        # Result: (num_nodes, n_streams, hidden_dim)
        residual = torch.bmm(H_res, x_streams)  # batch matrix multiplication

        # ============================================================
        # Message passing path: (H^post)^T * message_output
        # Expands single GNN output to all n streams with learned weights
        # ============================================================
        # message_output: (num_nodes, hidden_dim) - single vector per node
        # H_post: (num_nodes, 1, n_streams)
        # H_post^T: (num_nodes, n_streams, 1)
        # Broadcasting expands message to all streams
        message_expanded = H_post.transpose(1, 2) * message_output.unsqueeze(1)
        # Result: (num_nodes, n_streams, hidden_dim)

        # ============================================================
        # Combine paths (Paper Eq. 15)
        # ============================================================
        x_streams_new = residual + message_expanded

        return x_streams_new, H_pre

    def extra_repr(self):
        return (f'n_streams={self.n_streams}, hidden_dim={self.hidden_dim}, '
                f'tau={self.sinkhorn_tau}, iters={self.sinkhorn_iters}')


if __name__ == "__main__":
    # Test HyperConnection layer
    print("Testing HyperConnection layer...")

    n_streams = 4
    hidden_dim = 64
    num_nodes = 100

    hc = HyperConnection(
        n_streams=n_streams,
        hidden_dim=hidden_dim,
        sinkhorn_tau=0.1,
        sinkhorn_iters=10,
    )

    # Dummy inputs
    x_streams = torch.randn(num_nodes, n_streams, hidden_dim)
    message_output = torch.randn(num_nodes, hidden_dim)

    # Forward pass
    x_new, H_pre = hc(x_streams, message_output)

    print(f"Input shape: {x_streams.shape}")
    print(f"Message shape: {message_output.shape}")
    print(f"Output shape: {x_new.shape}")
    print(f"H_pre shape: {H_pre.shape}")

    # Test gradient
    loss = x_new.sum()
    loss.backward()
    print(f"\nGradients computed successfully")
    print(f"theta_pre grad norm: {hc.theta_pre.grad.norm().item():.6f}")

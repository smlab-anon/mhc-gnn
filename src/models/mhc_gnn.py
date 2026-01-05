"""
mHC-GNN: Manifold-Constrained Hyper-Connections for Graph Neural Networks

This is the main model implementing mHC-GNN from the paper. The architecture
wraps any base GNN (GCN, GAT, GraphSAGE, GIN) with multi-stream representations
and manifold-constrained stream mixing.

ARCHITECTURE OVERVIEW (Paper Section 4):
---------------------------------------
1. Input Expansion: x ∈ R^d → X ∈ R^{n×d} (n parallel streams)
2. For each layer l:
   a. Aggregate streams for GNN input (currently: mean pooling)
   b. Apply base GNN message passing: F_GNN(aggregated, neighbors)
   c. Apply hyper-connection update (Paper Eq. 15):
      X^{l+1} = H^res · X^{l} + (H^post)^T · F_GNN(...)
   where H^res ∈ Birkhoff polytope (doubly stochastic)
3. Final Readout: aggregate streams → classifier

THEORETICAL GUARANTEES:
----------------------
- Theorem 1: Over-smoothing rate (1-γ)^{L/n} vs (1-γ)^L for standard GNNs
- Theorem 2: Expressiveness beyond 1-WL through multi-stream representations

IMPLEMENTATION NOTE:
-------------------
The paper (Eq. 15) specifies aggregating streams via H^pre before GNN:
    F_GNN(H^pre · X^{l}, neighbors)

Current implementation uses mean pooling for simplicity:
    F_GNN(mean(X^{l}), neighbors)

This is equivalent to H^pre = [1/n, ..., 1/n] (uniform weights).
The proofs do NOT depend on H^pre, so this simplification is valid.

TO MATCH PAPER EXACTLY:
    x_aggregated = torch.bmm(H_pre, x_streams).squeeze(1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from layers.hyper_connections import HyperConnection
except ImportError:
    # Try relative import
    from ..layers.hyper_connections import HyperConnection


class mHCGNN(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Graph Neural Network.

    Supports multiple GNN backbones (GCN, GAT, GraphSAGE, GIN) with multi-stream architecture.
    
    Key components:
        - n_streams parallel representations per node
        - Doubly stochastic mixing matrices H^res (via Sinkhorn projection)
        - Architecture-agnostic: wraps any base GNN
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension d per stream
        out_channels: Output dimension (number of classes)
        num_layers: Number of GNN + hyper-connection layers L
        n_streams: Number of parallel streams n (expansion rate)
        gnn_type: Base GNN architecture ('gcn', 'gat', 'sage', 'gin')
        dropout: Dropout rate
        sinkhorn_tau: Temperature for Sinkhorn projection
        sinkhorn_iters: Number of Sinkhorn iterations
        use_dynamic: Use input-dependent mapping components
        use_static: Use learnable static biases
        task: 'node' for node classification, 'graph' for graph classification
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        n_streams=4,
        gnn_type='gcn',
        dropout=0.5,
        sinkhorn_tau=0.1,
        sinkhorn_iters=10,
        use_dynamic=True,
        use_static=True,
        task='node',  # 'node' or 'graph'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.n_streams = n_streams
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.task = task

        # ============================================================
        # Initial expansion: x ∈ R^d → X ∈ R^{n×d}
        # Maps input features to n parallel streams
        # ============================================================
        self.expand = nn.Linear(in_channels, n_streams * hidden_channels)

        # GNN layers and hyper-connections
        self.gnn_layers = nn.ModuleList()
        self.hyper_connections = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            # ============================================================
            # Base GNN layer (F_GNN in paper)
            # Operates on aggregated stream representation
            # ============================================================
            if self.gnn_type == 'gcn':
                gnn = GCNConv(hidden_channels, hidden_channels)
            elif self.gnn_type == 'gat':
                gnn = GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
            elif self.gnn_type == 'sage':
                gnn = SAGEConv(hidden_channels, hidden_channels)
            elif self.gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                gnn = GINConv(mlp)
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

            self.gnn_layers.append(gnn)

            # ============================================================
            # Hyper-connection layer
            # Computes H^pre, H^post, H^res and applies stream update
            # H^res is projected onto Birkhoff polytope (doubly stochastic)
            # ============================================================
            hc = HyperConnection(
                n_streams=n_streams,
                hidden_dim=hidden_channels,
                sinkhorn_tau=sinkhorn_tau,
                sinkhorn_iters=sinkhorn_iters,
                use_dynamic=use_dynamic,
                use_static=use_static,
            )
            self.hyper_connections.append(hc)

            # Batch normalization for stability
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Final readout: streams → output logits
        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Implements the full mHC-GNN forward propagation (Paper Eq. 15, 22):
            X^{l+1} = H^res · X^{l} + (H^post)^T · F_GNN(agg(X^{l}), neighbors)
        
        For L layers, final representation (Paper Eq. 22):
            X^{L} = (∏_{k=0}^{L-1} H^res_k) · X^{0} + Σ_{k=0}^{L-1} (∏_{j=k+1}^{L-1} H^res_j) · (H^post_k)^T · F_GNN^{k}

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            batch: Batch assignment for graph-level tasks (num_nodes,)

        Returns:
            Output logits (num_nodes, out_channels) for node tasks
            or (num_graphs, out_channels) for graph tasks
        """
        num_nodes = x.size(0)

        # ============================================================
        # Step 1: Initial expansion to multi-stream representation
        # x ∈ R^{in_channels} → X ∈ R^{n_streams × hidden_channels}
        # ============================================================
        x_expanded = self.expand(x)  # (num_nodes, n_streams * hidden_channels)
        x_streams = x_expanded.view(num_nodes, self.n_streams, self.hidden_channels)
        # Shape: (num_nodes, n_streams, hidden_channels)

        # Store for analysis (optional)
        self.stream_representations = []

        # ============================================================
        # Step 2: Process through L layers
        # Each layer: GNN message passing + hyper-connection update
        # ============================================================
        for layer_idx in range(self.num_layers):
            # Get current layer's modules
            gnn = self.gnn_layers[layer_idx]
            hc = self.hyper_connections[layer_idx]
            bn = self.batch_norms[layer_idx]

            # --------------------------------------------------------
            # Step 2a: Aggregate streams for GNN input
            # 
            # PAPER (Eq. 15): Uses H^pre for weighted aggregation
            #     x_aggregated = H^pre @ X  where H^pre ∈ R^{1×n}
            #
            # IMPLEMENTATION: Uses mean pooling (simpler, equivalent to uniform H^pre)
            #     x_aggregated = mean(X, dim=streams)
            #
            # TO MATCH PAPER:
            #     H_pre, _, _ = hc.compute_mappings(x_streams)
            #     x_aggregated = torch.bmm(H_pre, x_streams).squeeze(1)
            # --------------------------------------------------------
            # Compute H^pre (returned but currently unused for aggregation)
            _, _, H_pre_temp = hc.compute_mappings(x_streams)
            
            # Mean aggregation across streams (simpler than learned H^pre)
            x_aggregated = x_streams.mean(dim=1)  # (num_nodes, hidden_channels)

            # --------------------------------------------------------
            # Step 2b: GNN message passing F_GNN
            # This is the base GNN (GCN/GAT/SAGE/GIN) operating on 
            # the aggregated single-vector representation
            # --------------------------------------------------------
            x_msg = gnn(x_aggregated, edge_index)  # (num_nodes, hidden_channels)
            x_msg = F.relu(x_msg)
            x_msg = F.dropout(x_msg, p=self.dropout, training=self.training)

            # Apply batch normalization for training stability
            x_msg = bn(x_msg)

            # --------------------------------------------------------
            # Step 2c: Hyper-connection update (Paper Eq. 15)
            # X^{l+1} = H^res · X^{l} + (H^post)^T · x_msg
            #
            # H^res ∈ Birkhoff polytope (doubly stochastic) - THIS IS CRITICAL
            # Ensures slower over-smoothing rate per Theorem 1
            # --------------------------------------------------------
            x_streams, H_pre = hc(x_streams, x_msg)

            # Store for analysis
            self.stream_representations.append(x_streams.detach())

        # ============================================================
        # Step 3: Final readout
        # Aggregate n streams back to single representation → classifier
        # ============================================================
        x_final = x_streams.mean(dim=1)  # (num_nodes, hidden_channels)

        # Task-specific output
        if self.task == 'node':
            # Node-level prediction
            out = self.readout(x_final)  # (num_nodes, out_channels)
        elif self.task == 'graph':
            # Graph-level prediction
            if batch is None:
                # Single graph: global mean pooling
                x_graph = x_final.mean(dim=0, keepdim=True)  # (1, hidden_channels)
            else:
                # Multiple graphs: batch-wise mean pooling
                from torch_geometric.nn import global_mean_pool
                x_graph = global_mean_pool(x_final, batch)  # (num_graphs, hidden_channels)
            out = self.readout(x_graph)  # (num_graphs, out_channels)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        return out

    def get_stream_representations(self):
        """Get stored stream representations for analysis."""
        return self.stream_representations


class StandardGNN(nn.Module):
    """
    Standard GNN baseline (no hyper-connections) for comparison.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        gnn_type='gcn',
        dropout=0.5,
        task='node',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.task = task

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        if gnn_type == 'gcn':
            self.layers.append(GCNConv(in_channels, hidden_channels))
        elif gnn_type == 'gat':
            self.layers.append(GATConv(in_channels, hidden_channels, heads=1))
        elif gnn_type == 'sage':
            self.layers.append(SAGEConv(in_channels, hidden_channels))
        elif gnn_type == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.layers.append(GINConv(mlp))

        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            if gnn_type == 'gcn':
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            elif gnn_type == 'gat':
                self.layers.append(GATConv(hidden_channels, hidden_channels, heads=1))
            elif gnn_type == 'sage':
                self.layers.append(SAGEConv(hidden_channels, hidden_channels))
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.layers.append(GINConv(mlp))

            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == 'graph' and batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)

        return self.readout(x)


if __name__ == "__main__":
    # Test mHC-GNN
    print("Testing mHC-GNN...")

    from torch_geometric.data import Data

    # Create dummy graph
    num_nodes = 100
    num_edges = 300
    in_channels = 16
    hidden_channels = 64
    out_channels = 7
    num_layers = 4

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Test node classification
    print("\n=== Node Classification ===")
    model = mHCGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        n_streams=4,
        gnn_type='gcn',
        task='node',
    )

    out = model(x, edge_index)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (num_nodes, out_channels)

    # Test gradient
    loss = out.sum()
    loss.backward()
    print("Gradient computed successfully")

    # Test graph classification
    print("\n=== Graph Classification ===")
    batch = torch.zeros(num_nodes, dtype=torch.long)
    batch[50:] = 1  # 2 graphs

    model_graph = mHCGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        n_streams=4,
        gnn_type='gcn',
        task='graph',
    )

    out_graph = model_graph(x, edge_index, batch)
    print(f"Output: {out_graph.shape}")
    assert out_graph.shape == (2, out_channels)  # 2 graphs

    print("\n=== Baseline GNN ===")
    baseline = StandardGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        gnn_type='gcn',
        task='node',
    )

    out_baseline = baseline(x, edge_index)
    print(f"Baseline output: {out_baseline.shape}")

    print("\n✓ All tests passed!")

# mHC-GNN: Manifold-Constrained Hyper-Connections for Graph Neural Networks

Official implementation of **"Manifold-Constrained Hyper-Connections for Graph Neural Networks"**.

## Overview

mHC-GNN addresses the over-smoothing problem in deep Graph Neural Networks (GNNs) through manifold-constrained hyper-connections. Our method enables training networks exceeding 100 layers while maintaining strong performance.

### Key Features

- **Unprecedented depth**: Successfully trains GNNs with up to **128 layers**
- **Dramatic improvements**: +50% absolute accuracy improvement at extreme depths
- **Architecture-agnostic**: Works with GCN, GraphSAGE, GAT, and GIN
- **Theoretically grounded**: $(1-\gamma)^{L/n}$ convergence rate vs standard $(1-\gamma)^L$

### Main Results

At 128 layers on citation networks:
- **Cora**: Baseline 21.6% → mHC-GNN 73.4% (**+51.8%**)
- **CiteSeer**: Baseline 19.9% → mHC-GNN 58.8% (**+39.0%**)
- **PubMed**: Baseline 39.8% → mHC-GNN 74.5% (**+34.7%**)

Standard GCN collapses to near-random performance beyond 16 layers, while mHC-GNN maintains strong accuracy even at unprecedented depths.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+
- CUDA 11.3+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone 
cd mhc-gnn

# Create conda environment
conda create -n mhc-gnn python=3.10
conda activate mhc-gnn

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse
```

## Quick Start

### Training a Model

```bash
# Train mHC-GNN with 8 layers on Cora
python src/experiments/run_node_classification.py \
    --dataset Cora \
    --model mhc_gnn \
    --gnn_type gcn \
    --num_layers 8 \
    --hidden_channels 128 \
    --n_streams 4 \
    --epochs 500 \
    --seed 42

# Train baseline GCN for comparison
python src/experiments/run_node_classification.py \
    --dataset Cora \
    --model standard_gnn \
    --gnn_type gcn \
    --num_layers 8 \
    --hidden_channels 128 \
    --epochs 500 \
    --seed 42
```

### Running Complete Experiments

To reproduce all paper experiments:

```bash
# Run specific experiment suite
./run_all_experiments.sh main        # Main 8-layer benchmarks
./run_all_experiments.sh multi-arch  # Multi-architecture experiments
./run_all_experiments.sh depth       # Extended depth analysis (requires 4 GPUs)
./run_all_experiments.sh ablation    # Ablation studies

# Or run everything
./run_all_experiments.sh all
```

**Estimated time**:
- Main benchmarks: ~2 hours
- Multi-architecture: ~4 hours
- Depth analysis: ~5 hours (requires 4 GPUs)
- Ablation studies: ~1 hour

## Project Structure

```
mhc-gnn/
├── src/
│   ├── models/
│   │   └── mhc_gnn.py              # Main mHC-GNN model
│   ├── layers/
│   │   └── hyper_connections.py    # Hyper-connection layer with Birkhoff projection
│   ├── utils/
│   │   └── sinkhorn.py             # Sinkhorn-Knopp algorithm
│   └── experiments/
│       └── run_node_classification.py  # Training script
├── data/                           # Dataset directory (auto-downloaded)
├── results/                        # Experimental results
├── figures/                        # Generated figures
├── create_deep_depth_plots.py     # Visualization script
├── run_deep_depth_analysis.sh     # Depth analysis experiments
└── README.md
```

## Key Components

### mHC-GNN Model

The core model implements manifold-constrained hyper-connections:

```python
from models.mhc_gnn import mHCGNN

model = mHCGNN(
    in_channels=dataset.num_features,
    hidden_channels=128,
    out_channels=dataset.num_classes,
    num_layers=32,           # Deep network!
    n_streams=4,             # Number of parallel streams
    gnn_type='gcn',          # Base architecture
    use_dynamic=True,        # Dynamic hyper-connections
    use_static=True,         # Static hyper-connections
    sinkhorn_tau=0.1,        # Temperature for Sinkhorn
    sinkhorn_iters=20        # Sinkhorn iterations
)
```

### Hyper-Connection Layer

Each layer computes:
1. **Stream mixing**: Projects connectivity matrices onto Birkhoff polytope
2. **Message passing**: Applies base GNN within each stream
3. **Residual connection**: Maintains information flow

```python
x_streams, H = hyper_connection(x_streams, x_msg)
```

### Birkhoff Projection

Uses differentiable Sinkhorn-Knopp algorithm to project matrices onto the space of doubly-stochastic matrices:

```python
from utils.sinkhorn import sinkhorn_knopp

H_birkhoff = sinkhorn_knopp(H, tau=0.1, num_iters=20)
```

## Experimental Results

mHC-GNN demonstrates consistent improvements across diverse datasets and unprecedented depth capabilities:

- **Standard depth (8L)**: +2-9% on homophilic graphs, +6-8% on heterophilic graphs
- **Extreme depth (128L)**: Baseline collapses to 20-40% accuracy, mHC-GNN maintains 59-74%

See [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md) for complete experimental results and [docs/EXTENDED_DEPTH_EXPERIMENTS.md](docs/EXTENDED_DEPTH_EXPERIMENTS.md) for methodology details.

## Hyperparameters

### Default Settings (8 Layers)

| Parameter          | Value | Description                           |
|-------------------|-------|---------------------------------------|
| `hidden_channels` | 128   | Hidden dimension                      |
| `num_layers`      | 8     | Network depth                         |
| `n_streams`       | 4     | Number of parallel streams            |
| `sinkhorn_tau`    | 0.1   | Sinkhorn temperature                  |
| `sinkhorn_iters`  | 20    | Sinkhorn iterations                   |
| `dropout`         | 0.5   | Dropout rate                          |
| `lr`              | 0.001 | Learning rate                         |
| `weight_decay`    | 5e-4  | L2 regularization                     |
| `epochs`          | 500   | Maximum training epochs               |
| `patience`        | 100   | Early stopping patience               |

### Deep Networks (32+ Layers)

For very deep networks, consider:
- Increasing hidden dimensions: 256-512
- More streams: n=4 or n=8
- Lower learning rate: 0.0005
- Gradient clipping: 1.0

## GPU Requirements

- **8 layers**: ~2-4 GB GPU memory
- **16 layers**: ~4-8 GB GPU memory
- **32 layers**: ~8-16 GB GPU memory
- **64 layers**: ~16-32 GB GPU memory
- **128 layers**: ~32-48 GB GPU memory

For 128-layer models with n=4 streams on large datasets (PubMed), you may need 48GB+ GPUs.

## Datasets

The code automatically downloads and processes:

**Citation Networks**:
- Cora (2,708 nodes, 7 classes)
- CiteSeer (3,327 nodes, 6 classes)
- PubMed (19,717 nodes, 3 classes)

**Heterophilic Networks**:
- Texas (183 nodes, 5 classes)
- Wisconsin (251 nodes, 5 classes)
- Cornell (183 nodes, 5 classes)
- Chameleon (2,277 nodes, 5 classes)
- Squirrel (5,201 nodes, 5 classes)
- Actor (7,600 nodes, 5 classes)

**Large-Scale**:
- ogbn-arxiv (169,343 nodes, 40 classes)

## Citation

If you use this code in your research, please cite:

```bibtex

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch Geometric team for the excellent GNN library
- Computing resources provided by National Institute of Science Education and Research (NISER)

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: smishra@niser.ac.in

## Additional Resources

- **Paper**: Available soon on arXiv
- **Extended results**: See [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md)
- **Experimental details**: See [docs/EXTENDED_DEPTH_EXPERIMENTS.md](docs/EXTENDED_DEPTH_EXPERIMENTS.md)
- **Visualizations**: Generated figures in `figures/` directory

---

**Note**: This is research code. For production use, consider additional error handling, logging, and validation.

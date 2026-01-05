"""
Experiment runner for node classification tasks.
Supports multi-GPU training on 4× A6000 Ada.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.loader import DataLoader
import argparse
import yaml
import wandb
import os
import sys
import numpy as np
import random
from pathlib import Path

# Import OGB datasets
try:
    from ogb.nodeproppred import PygNodePropPredDataset
    OGB_AVAILABLE = True

    # Workaround for PyTorch 2.6+ weights_only=True default
    # OGB datasets need weights_only=False
    import torch._utils
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

except ImportError:
    OGB_AVAILABLE = False
    print("Warning: ogb not installed. Large-scale datasets unavailable.")

# Fix import paths - add src directory to Python path
# This allows running from project root: python src/experiments/run_node_classification.py
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent  # Go up from experiments/ to src/
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.mhc_gnn import mHCGNN, StandardGNN
from utils.trainer import Trainer, setup_distributed, cleanup_distributed, EarlyStopping
from data.heterophilic_datasets import load_heterophilic_dataset


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(dataset_name, root='./data'):
    """Load dataset."""
    dataset_name_lower = dataset_name.lower()

    # Homophilic datasets (citation networks)
    if dataset_name_lower in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=dataset_name)

    # Heterophilic datasets
    elif dataset_name_lower in ['chameleon', 'squirrel', 'actor', 'cornell', 'texas', 'wisconsin']:
        dataset = load_heterophilic_dataset(dataset_name_lower, root=root)

    # OGB large-scale datasets
    elif dataset_name_lower in ['ogbn-arxiv', 'ogbn-products']:
        if not OGB_AVAILABLE:
            raise ImportError("ogb not installed. Run: pip install ogb")
        dataset = PygNodePropPredDataset(name=dataset_name_lower, root=root)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def create_model(args, in_channels, out_channels):
    """Create model based on args."""
    if args.model == 'mhc_gnn':
        model = mHCGNN(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            num_layers=args.num_layers,
            n_streams=args.n_streams,
            gnn_type=args.gnn_type,
            dropout=args.dropout,
            sinkhorn_tau=args.sinkhorn_tau,
            sinkhorn_iters=args.sinkhorn_iters,
            use_dynamic=args.use_dynamic,
            use_static=args.use_static,
            task='node',
        )
    elif args.model == 'standard_gnn':
        model = StandardGNN(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            num_layers=args.num_layers,
            gnn_type=args.gnn_type,
            dropout=args.dropout,
            task='node',
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


def train(rank, world_size, args):
    """Training function for single or multi-GPU."""
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)

    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Set seed
    set_seed(args.seed + rank)  # Different seed per process

    # Load dataset
    dataset = load_dataset(args.dataset, root=args.data_root)
    data = dataset[0].to(device)

    # Handle OGB datasets (different format)
    if args.dataset.lower().startswith('ogbn-'):
        split_idx_dict = dataset.get_idx_split()

        # OGB datasets don't have masks, they have index arrays
        # Convert to masks for consistency
        num_nodes = data.num_nodes
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[split_idx_dict['train']] = True
        data.val_mask[split_idx_dict['valid']] = True
        data.test_mask[split_idx_dict['test']] = True

        # OGB datasets have labels as (num_nodes, 1), need to squeeze
        if data.y.dim() == 2:
            data.y = data.y.squeeze(1)

        if rank == 0:
            print(f"OGB dataset loaded: {num_nodes} nodes")
            print(f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")

    # Handle heterophilic datasets with multiple splits
    # These datasets have train_mask with shape [num_nodes, num_splits]
    elif hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
        # Select split based on seed (use split_idx = seed % num_splits)
        split_idx = args.split_idx if hasattr(args, 'split_idx') else 0
        if rank == 0:
            print(f"Using split {split_idx} out of {data.train_mask.shape[1]} available splits")

        data.train_mask = data.train_mask[:, split_idx]
        data.val_mask = data.val_mask[:, split_idx]
        data.test_mask = data.test_mask[:, split_idx]

    # Create model
    model = create_model(args, dataset.num_features, dataset.num_classes)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        task='node',
        scheduler=scheduler,
        use_amp=args.use_amp,
        world_size=world_size,
        rank=rank,
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')

    # Initialize wandb (only on main process)
    if rank == 0 and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=vars(args),
        )

    # Training loop
    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=-1)

            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

            # Save checkpoint
            if rank == 0:
                trainer.save_checkpoint(
                    os.path.join(args.save_dir, f'{args.exp_name}_best.pt'),
                    epoch,
                    best_val_acc
                )

        # Logging
        if rank == 0 and (epoch % args.log_every == 0 or epoch == args.epochs):
            print(f'Epoch {epoch:03d}: Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'best_val_acc': best_val_acc,
                    'best_test_acc': best_test_acc,
                })

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_acc)

        # Early stopping
        if early_stopping(val_acc):
            if rank == 0:
                print(f"Early stopping at epoch {epoch}")
            break

    # Final results
    if rank == 0:
        print(f"\nBest Val Acc: {best_val_acc:.4f}")
        print(f"Test Acc at Best Val: {best_test_acc:.4f}")

        if args.use_wandb:
            wandb.log({
                'final/best_val_acc': best_val_acc,
                'final/test_acc': best_test_acc,
            })
            wandb.finish()

    # Cleanup
    if world_size > 1:
        cleanup_distributed()

    return best_val_acc, best_test_acc


def main():
    parser = argparse.ArgumentParser(description='mHC-GNN Node Classification')

    # Dataset
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed',  # Homophilic
                                 'chameleon', 'squirrel', 'actor',  # Heterophilic
                                 'texas', 'wisconsin', 'cornell',  # Heterophilic (small)
                                 'ogbn-arxiv', 'ogbn-products'])  # Large-scale OGB
    parser.add_argument('--data_root', type=str, default='./data')

    # Model
    parser.add_argument('--model', type=str, default='mhc_gnn', choices=['mhc_gnn', 'standard_gnn'])
    parser.add_argument('--gnn_type', type=str, default='gcn', choices=['gcn', 'gat', 'sage', 'gin'])
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)

    # mHC-specific
    parser.add_argument('--n_streams', type=int, default=4)
    parser.add_argument('--sinkhorn_tau', type=float, default=0.1)
    parser.add_argument('--sinkhorn_iters', type=int, default=10)
    parser.add_argument('--use_dynamic', action='store_true', default=True)
    parser.add_argument('--use_static', action='store_true', default=True)

    # Training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--use_amp', action='store_true', default=False)

    # Multi-GPU
    parser.add_argument('--num_gpus', type=int, default=1)

    # Experiment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_idx', type=int, default=0, help='Split index for heterophilic datasets (0-9)')
    parser.add_argument('--exp_name', type=str, default='mhc_gnn_cora')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_every', type=int, default=10, help='Log every N epochs')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='mhc-gnn')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Print configuration
    print("=" * 50)
    print("Experiment Configuration")
    print("=" * 50)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 50)

    # Run training
    if args.num_gpus > 1:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)
    else:
        train(0, 1, args)


if __name__ == '__main__':
    main()

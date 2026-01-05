"""
Training utilities for mHC-GNN with multi-GPU support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


class Trainer:
    """
    Trainer for GNN models with multi-GPU support.
    """
    def __init__(
        self,
        model,
        optimizer,
        device,
        task='node',
        loss_fn=None,
        scheduler=None,
        use_amp=False,
        world_size=1,
        rank=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.task = task
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.world_size = world_size
        self.rank = rank

        # Loss function
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        # AMP scaler
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # Move model to device
        self.model = self.model.to(device)

        # Wrap with DDP if multi-GPU
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training", disable=(self.rank != 0))
        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = self._compute_loss(out, batch)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self._compute_loss(out, batch)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, loader, return_preds=False):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self._compute_loss(out, batch)

                total_loss += loss.item()
                num_batches += 1

                # Get predictions
                if self.task in ['node', 'graph']:
                    preds = out.argmax(dim=-1)
                else:
                    preds = torch.sigmoid(out)

                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())

        avg_loss = total_loss / num_batches

        # Concatenate predictions
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics['loss'] = avg_loss

        if return_preds:
            return metrics, all_preds, all_labels
        return metrics

    def _compute_loss(self, out, batch):
        """Compute loss based on task."""
        if hasattr(batch, 'train_mask'):
            # Node classification with mask
            return self.loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
        else:
            # Graph classification or full batch
            return self.loss_fn(out, batch.y)

    def _compute_metrics(self, preds, labels):
        """Compute evaluation metrics."""
        metrics = {}

        if self.task in ['node', 'graph']:
            # Classification
            acc = accuracy_score(labels.numpy(), preds.numpy())
            metrics['accuracy'] = acc
        else:
            # Regression or multi-label
            # Compute MAE, RMSE, etc.
            pass

        return metrics

    def save_checkpoint(self, path, epoch, best_val_metric):
        """Save model checkpoint."""
        if self.rank == 0:  # Only save on main process
            model_state = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_metric': best_val_metric,
            }

            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint.get('epoch', 0), checkpoint.get('best_val_metric', 0)


def setup_distributed(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience=20, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return False

        if self.mode == 'max':
            improved = metric > self.best_score + self.min_delta
        else:
            improved = metric < self.best_score - self.min_delta

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

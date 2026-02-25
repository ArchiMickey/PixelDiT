"""
Utility functions for DiT and PixelDiT models.

This module provides factory functions for creating models and datasets,
as well as checkpoint management utilities.
"""

import os
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pixel_dit.model import DiT, PixelDiT, MMDiT, MMPixelDiT
from pixel_dit.config import ModelConfig, DatasetConfig


def create_model(
    model_type: str,
    dataset_config: DatasetConfig,
    model_config: Optional[ModelConfig] = None,
) -> nn.Module:
    """
    Factory function to create a DiT, PixelDiT, MMDiT or MMPixelDiT model.
    
    Args:
        model_type: 'dit', 'pixeldit', 'mmdit' or 'mmpixeldit'
        dataset_config: Dataset configuration containing image size, channels, etc.
        model_config: Optional model configuration. If None, uses defaults.
    
    Returns:
        Model instance
    
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()
    
    if model_type == 'dit':
        if model_config is None:
            # Default DiT config
            model = DiT(
                image_size=dataset_config.image_size,
                patch_size=4,
                channels=dataset_config.in_channels,
                dim=128,
                depth=6,
                heads=4,
                dim_head=32,
                dropout=0.0,
                num_classes=dataset_config.num_classes,
                class_dropout_prob=0.1,
                qk_norm='rmsnorm'
            )
        else:
            model = DiT(
                image_size=dataset_config.image_size,
                patch_size=model_config.patch_size,
                channels=dataset_config.in_channels,
                dim=model_config.dim,
                depth=model_config.depth,
                heads=model_config.heads,
                dim_head=model_config.dim_head,
                mlp_ratio=model_config.mlp_ratio,
                dropout=model_config.dropout,
                num_classes=dataset_config.num_classes,
                class_dropout_prob=model_config.class_dropout_prob,
                use_abs_pe=model_config.use_abs_pe,
                qk_norm=model_config.qk_norm
            )
    elif model_type == 'pixeldit':
        if model_config is None:
            # Default PixelDiT config
            model = PixelDiT(
                image_size=dataset_config.image_size,
                semantic_patch_size=4,
                channels=dataset_config.in_channels,
                dit_dim=128,
                pit_dim=16,
                dit_depth=6,
                pit_depth=2,
                heads=4,
                dim_head=32,
                mlp_ratio=4.0,
                dropout=0.0,
                num_classes=dataset_config.num_classes,
                class_dropout_prob=0.1,
                compress_ratio=4,
                use_abs_pe=True,
                qk_norm='rmsnorm'
            )
        else:
            model = PixelDiT(
                image_size=dataset_config.image_size,
                semantic_patch_size=model_config.patch_size,
                channels=dataset_config.in_channels,
                dit_dim=model_config.dit_dim,
                pit_dim=model_config.pit_dim,
                dit_depth=model_config.dit_depth,
                pit_depth=model_config.pit_depth,
                heads=model_config.heads,
                dim_head=model_config.dim_head,
                mlp_ratio=model_config.mlp_ratio,
                dropout=model_config.dropout,
                num_classes=dataset_config.num_classes,
                class_dropout_prob=model_config.class_dropout_prob,
                compress_ratio=model_config.compress_ratio,
                use_abs_pe=model_config.use_abs_pe,
                qk_norm=model_config.qk_norm
            )
    elif model_type == 'mmdit':
        if model_config is None:
            # Default MMDiT config
            model = MMDiT(
                image_size=dataset_config.image_size,
                patch_size=4,
                channels=dataset_config.in_channels,
                dim=128,
                depth=6,
                heads=4,
                dim_head=32,
                dropout=0.0,
                num_classes=dataset_config.num_classes,
                class_dropout_prob=0.1,
                qk_norm='rmsnorm'
            )
        else:
            model = MMDiT(
                image_size=dataset_config.image_size,
                patch_size=model_config.patch_size,
                channels=dataset_config.in_channels,
                dim=model_config.dim,
                depth=model_config.depth,
                heads=model_config.heads,
                dim_head=model_config.dim_head,
                mlp_ratio=model_config.mlp_ratio,
                dropout=model_config.dropout,
                num_classes=dataset_config.num_classes,
                num_registers=model_config.num_registers,
                class_dropout_prob=model_config.class_dropout_prob,
                use_abs_pe=model_config.use_abs_pe,
                qk_norm=model_config.qk_norm
            )
    elif model_type == 'mmpixeldit':
        if model_config is None:
            # Default MMPixelDiT config
            model = MMPixelDiT(
                image_size=dataset_config.image_size,
                semantic_patch_size=4,
                channels=dataset_config.in_channels,
                dit_dim=128,
                pit_dim=16,
                dit_depth=6,
                pit_depth=2,
                heads=4,
                dim_head=32,
                mlp_ratio=4.0,
                dropout=0.0,
                num_classes=dataset_config.num_classes,
                class_dropout_prob=0.1,
                compress_ratio=4,
                use_abs_pe=True,
                qk_norm='rmsnorm'
            )
        else:
            model = MMPixelDiT(
                image_size=dataset_config.image_size,
                semantic_patch_size=model_config.patch_size,
                channels=dataset_config.in_channels,
                dit_dim=model_config.dit_dim,
                pit_dim=model_config.pit_dim,
                dit_depth=model_config.dit_depth,
                pit_depth=model_config.pit_depth,
                heads=model_config.heads,
                dim_head=model_config.dim_head,
                mlp_ratio=model_config.mlp_ratio,
                dropout=model_config.dropout,
                num_classes=dataset_config.num_classes,
                num_registers=model_config.num_registers,
                class_dropout_prob=model_config.class_dropout_prob,
                compress_ratio=model_config.compress_ratio,
                use_abs_pe=model_config.use_abs_pe,
                qk_norm=model_config.qk_norm,
            )
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: 'dit', 'pixeldit', 'mmdit', 'mmpixeldit'"
        )
    
    return model


def get_dataset(
    dataset_name: str,
    root: str = './data',
    dataset_config: Optional[DatasetConfig] = None,
) -> DataLoader:
    """
    Get training DataLoader for MNIST, CIFAR10 or ImageNet.
    
    Args:
        dataset_name: 'mnist', 'cifar10' or 'imagenet'
        root: Root directory for data storage
        dataset_config: Optional dataset configuration. If None, uses defaults.
    
    Returns:
        DataLoader for the training dataset
    
    Raises:
        ValueError: If dataset_name is not supported
    """
    dataset_name = dataset_name.lower()
    
    if dataset_config is None:
        if dataset_name == 'mnist':
            normalize_mean = (0.5,)
            normalize_std = (0.5,)
            image_size = 28
        elif dataset_name == 'cifar10':
            normalize_mean = (0.5, 0.5, 0.5)
            normalize_std = (0.5, 0.5, 0.5)
            image_size = 32
        elif dataset_name == 'imagenet':
            # Standard ImageNet normalization
            normalize_mean = (0.485, 0.456, 0.406)
            normalize_std = (0.229, 0.224, 0.225)
            image_size = 256
        else:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported datasets: 'mnist', 'cifar10', 'imagenet'"
            )
    else:
        normalize_mean = dataset_config.normalize_mean
        normalize_std = dataset_config.normalize_std
        image_size = dataset_config.image_size
    
    if dataset_name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name == 'imagenet':
        # Expecting ImageNet to be in root/train
        train_dir = os.path.join(root, 'train')
        if not os.path.exists(train_dir):
            # Fallback to root if 'train' subdir doesn't exist
            train_dir = root
        dataset = datasets.ImageFolder(
            root=train_dir,
            transform=transform
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: 'mnist', 'cifar10', 'imagenet'"
        )
    
    return dataset


def get_output_dirs(
    model_type: str,
    dataset_name: str,
    checkpoint_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Get checkpoint and results directories for a model-dataset combination.
    
    The directories are organized as:
    - checkpoints/{model}/{dataset}
    - results/{model}/{dataset}
    
    Args:
        model_type: 'dit' or 'pixeldit'
        dataset_name: 'mnist' or 'cifar10'
        checkpoint_dir: Override checkpoint directory (if provided, returned as-is)
        results_dir: Override results directory (if provided, returned as-is)
    
    Returns:
        Tuple of (checkpoint_dir, results_dir)
    """
    model_type = model_type.lower()
    dataset_name = dataset_name.lower()
    
    # If directories are explicitly provided, use them
    if checkpoint_dir is not None and results_dir is not None:
        return checkpoint_dir, results_dir
    
    # Default directories: checkpoints/{model}/{dataset}
    default_checkpoint_dir = os.path.join('checkpoints', model_type, dataset_name)
    default_results_dir = os.path.join('results', model_type, dataset_name)
    
    checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else default_checkpoint_dir
    results_dir = results_dir if results_dir is not None else default_results_dir
    
    return checkpoint_dir, results_dir


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_type: str,
    dataset_name: str,
    config: Optional[Dict[str, Any]] = None,
    checkpoint_dir: str = 'checkpoints',
    loss: Optional[float] = None,
    ema_model: Optional[nn.Module] = None,
) -> str:
    """
    Save a training checkpoint with metadata.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        model_type: 'dit', 'pixeldit', 'mmdit' or 'mmpixeldit'
        dataset_name: 'mnist' or 'cifar10'
        config: Optional configuration dictionary to save
        checkpoint_dir: Directory to save checkpoint
        loss: Optional loss value to save
        ema_model: Optional EMA model to save
    
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Backward-compatible checkpoint naming
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{model_type}_{dataset_name}_epoch_{epoch}.pt"
    )
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_type': model_type,
        'dataset_name': dataset_name,
    }
    
    if ema_model is not None:
        checkpoint_data['ema_model_state_dict'] = ema_model.state_dict()
    
    if loss is not None:
        checkpoint_data['loss'] = loss
    
    if config is not None:
        checkpoint_data['config'] = config
    
    torch.save(checkpoint_data, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    ema_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint into a model.
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint to (default: same as model)
        ema_model: Optional EMA model to load weights into
    
    Returns:
        Dictionary containing checkpoint metadata (epoch, model_type, dataset_name, etc.)
    """
    if device is None:
        device = next(model.parameters()).device
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if ema_model is not None and 'ema_model_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'model_type': checkpoint.get('model_type', None),
        'dataset_name': checkpoint.get('dataset_name', None),
        'loss': checkpoint.get('loss', None),
        'config': checkpoint.get('config', None),
    }
    
    return metadata


def detect_model_type_from_checkpoint(path: str) -> Optional[str]:
    """
    Detect model type from checkpoint filename or contents.
    
    Args:
        path: Path to checkpoint file
    
    Returns:
        Model type ('dit', 'pixeldit', 'mmdit' or 'mmpixeldit') or None if cannot detect
    """
    # Try to detect from filename
    filename = os.path.basename(path)
    if filename.startswith('dit_'):
        return 'dit'
    elif filename.startswith('pixeldit_'):
        return 'pixeldit'
    elif filename.startswith('mmdit_'):
        return 'mmdit'
    elif filename.startswith('mmpixeldit_'):
        return 'mmpixeldit'
    
    # Try to load and check metadata
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_type' in checkpoint:
            return checkpoint['model_type']
    except Exception:
        pass
    
    return None


def detect_dataset_from_checkpoint(path: str) -> Optional[str]:
    """
    Detect dataset name from checkpoint filename or contents.
    
    Args:
        path: Path to checkpoint file
    
    Returns:
        Dataset name ('mnist' or 'cifar10') or None if cannot detect
    """
    # Try to detect from filename
    filename = os.path.basename(path)
    if '_mnist_' in filename:
        return 'mnist'
    elif '_cifar10_' in filename:
        return 'cifar10'
    
    # Try to load and check metadata
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if 'dataset_name' in checkpoint:
            return checkpoint['dataset_name']
    except Exception:
        pass
    
    return None

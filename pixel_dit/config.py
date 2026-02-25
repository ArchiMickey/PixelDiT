"""
Configuration module for DiT and PixelDiT models.

This module provides dataclasses for model, dataset, and training configurations,
along with factory functions to load configurations from YAML files.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Any, Type, TypeVar
import yaml
from pathlib import Path

T = TypeVar('T', bound='BaseConfig')

# Get the configs directory path (relative to this file)
_CONFIG_DIR = Path(__file__).parent.parent / "configs"


@dataclass
class BaseConfig:
    """Base configuration class with common methods."""
    
    @classmethod
    def from_yaml(cls: Type[T], path: str) -> T:
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle special cases for DatasetConfig (tuples)
        if cls.__name__ == 'DatasetConfig':
            if 'normalize_mean' in data:
                data['normalize_mean'] = tuple(data['normalize_mean'])
            if 'normalize_std' in data:
                data['normalize_std'] = tuple(data['normalize_std'])
                
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for DiT/PixelDiT/MMDiT/MMPixelDiT models."""
    # Architecture type
    model_type: str = 'dit'  # 'dit', 'pixeldit', 'mmdit', or 'mmpixeldit'
    
    # Image parameters
    image_size: int = 28
    patch_size: int = 4  # For DiT/MMDiT; semantic_patch_size for PixelDiT/MMPixelDiT
    channels: int = 1
    
    # DiT/MMDiT-specific parameters
    dim: int = 128
    depth: int = 6
    heads: int = 4
    dim_head: int = 32
    mlp_ratio: float = 4.0
    qk_norm: str = None
    
    # PixelDiT/MMPixelDiT-specific parameters
    dit_dim: int = 128  # Semantic path dimension
    pit_dim: int = 16   # Pixel path dimension
    dit_depth: int = 6  # Semantic path depth
    pit_depth: int = 2  # Pixel path depth
    
    # MMDiT/MMPixelDiT-specific parameters
    num_registers: int = 4
    
    # Regularization
    dropout: float = 0.0
    
    # Class-conditional generation
    num_classes: int = 10
    class_dropout_prob: float = 0.1
    
    # PixelDiT/MMPixelDiT-specific
    compress_ratio: int = 4
    use_abs_pe: bool = True
    semantic_patch_size: Optional[int] = None  # If None, uses patch_size

    # RectifiedFlow parameters
    t_eps: float = 1e-5
    P_mean: float = 0.0
    P_std: float = 1.0


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for dataset."""
    name: str = 'mnist'
    image_size: int = 28
    in_channels: int = 1
    out_channels: int = 1
    num_classes: int = 10
    normalize_mean: Tuple[float, ...] = (0.5,)
    normalize_std: Tuple[float, ...] = (0.5,)


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training hyperparameters."""
    batch_size: int = 128
    lr: float = 1e-4
    epochs: int = 10
    sample_every: int = 500  # steps
    checkpoint_every: int = 1  # epochs
    num_workers: int = 2
    
    # Override directories (None means auto-determined)
    checkpoint_dir: Optional[str] = None
    results_dir: Optional[str] = None
    
    # Resume from checkpoint
    resume: Optional[str] = None


def get_model_config(model_type: str, dataset_name: str) -> ModelConfig:
    """
    Get model configuration for a specific model type and dataset.
    
    Args:
        model_type: 'dit', 'pixeldit', 'mmdit', or 'mmpixeldit'
        dataset_name: 'mnist' or 'cifar10'
    
    Returns:
        ModelConfig with appropriate settings
    
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = _CONFIG_DIR / "models" / f"{model_type.lower()}_{dataset_name.lower()}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Supported combinations: dit_mnist, dit_cifar10, pixeldit_mnist, pixeldit_cifar10, "
            f"mmdit_mnist, mmdit_cifar10, mmpixeldit_mnist, mmpixeldit_cifar10"
        )
    
    return ModelConfig.from_yaml(str(config_path))


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    Get dataset configuration for a specific dataset.
    
    Args:
        dataset_name: 'mnist' or 'cifar10'
    
    Returns:
        DatasetConfig with appropriate settings
    
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = _CONFIG_DIR / "datasets" / f"{dataset_name.lower()}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Supported datasets: mnist, cifar10"
        )
    
    return DatasetConfig.from_yaml(str(config_path))

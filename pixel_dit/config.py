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

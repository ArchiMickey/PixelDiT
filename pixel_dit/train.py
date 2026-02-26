"""
Unified training script for DiT and PixelDiT models.

This script supports training both DiT and PixelDiT models on MNIST or CIFAR10
datasets with configurable hyperparameters.

Usage:
    # Train DiT on MNIST (default)
    python -m pixel_dit.train --model dit --dataset mnist

    # Train PixelDiT on MNIST
    python -m pixel_dit.train --model pixeldit --dataset mnist

    # Train DiT on CIFAR10 with custom hyperparameters
    python -m pixel_dit.train --model dit --dataset cifar10 --epochs 20 --batch-size 64

    # Resume training from checkpoint
    python -m pixel_dit.train --model dit --dataset mnist --resume checkpoints/dit_mnist_epoch_5.pt
"""

import argparse
import os
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from accelerate import Accelerator

from pixel_dit.flow import RectifiedFlow
from pixel_dit.config import (
    DatasetConfig,
    TrainingConfig,
    get_dataset_config,
)
from pixel_dit.utils import (
    create_model,
    get_dataset,
    get_output_dirs,
    save_checkpoint,
    load_checkpoint,
)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA model parameters."""
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    
    for name, param in model_params.items():
        ema_params[name].copy_(
            ema_params[name] * decay + param.data * (1 - decay)
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified training script for DiT and PixelDiT models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model and dataset selection
    parser.add_argument(
        '--model',
        type=str,
        default='mmpixeldit-tiny/4',
        help='Model type and size to train (e.g., dit-tiny/4, mmdit-tiny/4, pixeldit-tiny/4, mmpixeldit-tiny/4)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'cifar10', 'imagenet'],
        default='mnist',
        help='Dataset to use for training',
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--resolution',
        type=int,
        default=None,
        help='Override training resolution (e.g., 64, 256)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for training',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--sample-every',
        type=int,
        default=500,
        help='Generate samples every N steps',
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=1,
        help='Save checkpoint every N epochs',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='Number of data loader workers',
    )
    
    # Output directories
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Checkpoint directory (auto-determined if None)',
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Results directory for samples (auto-determined if None)',
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from',
    )
    
    # EMA options
    parser.add_argument(
        '--no-ema',
        action='store_true',
        help='Disable EMA model and updates',
    )
    parser.add_argument(
        '--ema-decay',
        type=float,
        default=0.9999,
        help='EMA decay rate',
    )
    
    return parser.parse_args()


def train():
    """Main training function."""
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Get configurations
    dataset_config = get_dataset_config(args.dataset)
    
    # Override resolution if provided
    if args.resolution is not None:
        dataset_config.image_size = args.resolution
    
    # Get output directories
    checkpoint_dir, results_dir = get_output_dirs(
        args.model,
        args.dataset,
        args.checkpoint_dir,
        args.results_dir,
    )
    
    if accelerator.is_main_process:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = get_dataset(args.dataset, root='./data', dataset_config=dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    # Create model
    model = create_model(args.model, dataset_config)
    
    # Create EMA model
    ema_model = None
    if not args.no_ema:
        ema_model = copy.deepcopy(model)
        ema_model.requires_grad_(False)
        ema_model.eval()
    
    # Create RectifiedFlow wrapper
    rf = RectifiedFlow(
        model, 
        class_dropout_prob=0.1,
        t_eps=5e-2,
        P_mean=-0.8,
        P_std=0.8,
    )
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Prepare for acceleration
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if ema_model is not None:
        ema_model.to(device)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume is not None:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume}")
        metadata = load_checkpoint(
            args.resume, 
            accelerator.unwrap_model(model), 
            optimizer,
            ema_model=ema_model
        )
        start_epoch = metadata.get('epoch', 0) + 1
        if accelerator.is_main_process:
            print(f"Resuming from epoch {start_epoch}")
    
    if accelerator.is_main_process:
        print(f"Starting {args.model.upper()} training on {device}...")
        print(f"Dataset: {args.dataset}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"EMA enabled: {not args.no_ema}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Results directory: {results_dir}")
    
    step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{args.epochs}",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True
        )
        
        for batch_idx, (x, y) in enumerate(pbar):
            optimizer.zero_grad()
            loss = rf(x, y)
            accelerator.backward(loss)
            
            # Calculate gradient norm
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1000000.0)
            
            optimizer.step()
            
            # Update EMA
            if ema_model is not None:
                update_ema(ema_model, accelerator.unwrap_model(model), decay=args.ema_decay)
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                grad_norm=f"{grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm:.4f}"
            )
            
            # Generate samples
            if step % args.sample_every == 0:
                model.eval()
                if accelerator.is_main_process:
                    with torch.no_grad():
                        # Use EMA model for sampling if available, otherwise use base model
                        unwrapped_model = ema_model if ema_model is not None else accelerator.unwrap_model(model)
                        unwrapped_rf = RectifiedFlow(
                            unwrapped_model, 
                            class_dropout_prob=0.1,
                            t_eps=5e-2,
                            P_mean=-0.8,
                            P_std=0.8,
                        )
                        
                        # Sample with labels for visualization
                        # For large num_classes, sample a reasonable set of classes
                        if dataset_config.num_classes > 16:
                            y_sample = torch.linspace(0, dataset_config.num_classes - 1, 16, dtype=torch.long, device=device)
                        else:
                            y_sample = torch.arange(16, device=device) % dataset_config.num_classes
                            
                        samples = unwrapped_rf.sample(
                            shape=(16, dataset_config.in_channels, dataset_config.image_size, dataset_config.image_size),
                            steps=50,
                            device=device,
                            y=y_sample,
                        )
                        
                        # Denormalize using dataset-specific means/stds
                        # samples: [B, C, H, W]
                        means = torch.tensor(dataset_config.normalize_mean, device=device).view(1, -1, 1, 1)
                        stds = torch.tensor(dataset_config.normalize_std, device=device).view(1, -1, 1, 1)
                        samples = samples * stds + means
                        
                        save_image(samples, os.path.join(results_dir, f"sample_{step}.png"), nrow=4)
                model.train()
            
            step += 1
        
        # Save checkpoint
        if epoch % args.checkpoint_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_path = save_checkpoint(
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer,
                    epoch=epoch,
                    model_type=args.model,
                    dataset_name=args.dataset,
                    config={
                        'dataset': dataset_config.to_dict(),
                        'training': {
                            'batch_size': args.batch_size,
                            'lr': args.lr,
                            'epochs': args.epochs,
                        },
                    },
                    checkpoint_dir=checkpoint_dir,
                    loss=loss.item(),
                    ema_model=ema_model,
                )
                print(f"Saved checkpoint to {checkpoint_path}")
    
    if accelerator.is_main_process:
        print("Training complete.")


if __name__ == "__main__":
    train()

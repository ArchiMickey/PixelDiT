"""
Unified sampling script for DiT and PixelDiT models.

This script supports sampling from both DiT and PixelDiT models trained on
MNIST or CIFAR10 datasets. Model type and dataset can be auto-detected from
checkpoint metadata or filename.

Usage:
    # Sample from DiT trained on MNIST (auto-detected from checkpoint)
    python -m pixel_dit.sample --checkpoint checkpoints/dit_mnist_epoch_10.pt

    # Sample from PixelDiT with explicit model and dataset
    python -m pixel_dit.sample --checkpoint checkpoints_pixeldit/pixeldit_mnist_epoch_10.pt \\
        --model pixeldit --dataset mnist

    # Sample specific class label
    python -m pixel_dit.sample --checkpoint checkpoints/dit_mnist_epoch_10.pt \\
        --class-label 5 --num-samples 16

    # Sample with custom output path
    python -m pixel_dit.sample --checkpoint checkpoints/dit_mnist_epoch_10.pt \\
        --output my_samples.png --num-samples 100
"""

import argparse
import os
import math
from typing import Optional

import torch
from torchvision.utils import save_image, make_grid
from PIL import Image

from pixel_dit.flow import RectifiedFlow
from pixel_dit.config import get_dataset_config
from pixel_dit.utils import (
    create_model,
    get_output_dirs,
    detect_model_type_from_checkpoint,
    detect_dataset_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified sampling script for DiT and PixelDiT models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint',
    )
    
    # Model and dataset selection (auto-detected if not provided)
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model type and size (e.g., dit-tiny, mmdit-tiny) (auto-detected from checkpoint if None)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'cifar10', 'imagenet'],
        default=None,
        help='Dataset the model was trained on (auto-detected from checkpoint if None)',
    )
    
    # Sampling parameters
    parser.add_argument(
        '--num-samples',
        type=int,
        default=16,
        help='Number of samples to generate',
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=100,
        help='Number of ODE steps for sampling',
    )
    parser.add_argument(
        '--class-label',
        type=int,
        default=None,
        help='Specific class label to sample (None for random classes)',
    )
    parser.add_argument(
        '--cfg-scale',
        type=float,
        default=3.0,
        help='Classifier-free guidance scale',
    )
    parser.add_argument(
        '--cfg-interval',
        type=float,
        nargs=2,
        default=[0.1, 1.0],
        help='Time interval [start, end] to apply CFG',
    )
    parser.add_argument(
        '--shift',
        type=float,
        default=1.0,
        help='Time shift factor for sampling',
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for generated image (auto-determined if None)',
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Results directory (auto-determined if None)',
    )
    parser.add_argument(
        '--save-gif',
        action='store_true',
        default=False,
        help='Save a GIF of the sampling process',
    )
    parser.add_argument(
        '--gif-path',
        type=str,
        default=None,
        help='Output path for the GIF (auto-determined if None)',
    )
    
    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, auto-detected if None)',
    )
    
    return parser.parse_args()


def sample():
    """Main sampling function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect model type if not provided
    model_type = args.model
    if model_type is None:
        model_type = detect_model_type_from_checkpoint(args.checkpoint)
        if model_type is None:
            raise ValueError(
                f"Could not auto-detect model type from checkpoint: {args.checkpoint}. "
                f"Please specify --model (dit or pixeldit)."
            )
        print(f"Auto-detected model type: {model_type}")
    
    # Auto-detect dataset if not provided
    dataset_name = args.dataset
    if dataset_name is None:
        dataset_name = detect_dataset_from_checkpoint(args.checkpoint)
        if dataset_name is None:
            raise ValueError(
                f"Could not auto-detect dataset from checkpoint: {args.checkpoint}. "
                f"Please specify --dataset (mnist, cifar10, or imagenet)."
            )
        print(f"Auto-detected dataset: {dataset_name}")
    
    # Get configurations
    dataset_config = get_dataset_config(dataset_name)
    
    # Get output directories
    _, results_dir = get_output_dirs(
        model_type,
        dataset_name,
        results_dir=args.results_dir,
    )
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'generated_samples.png')
    
    # Determine GIF path
    gif_path = args.gif_path
    if gif_path is None and args.save_gif:
        os.makedirs(results_dir, exist_ok=True)
        gif_path = os.path.join(results_dir, 'sampling_process.gif')
    
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    print(f"Dataset: {dataset_name}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Create model
    model = create_model(model_type, dataset_config)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Determine which state dict to use
    if 'ema_model_state_dict' in checkpoint and checkpoint['ema_model_state_dict'] is not None:
        print("Using EMA weights for sampling.")
        state_dict = checkpoint['ema_model_state_dict']
    else:
        print("EMA weights not found in checkpoint. Falling back to base model weights.")
        state_dict = checkpoint['model_state_dict']
    
    # Handle state dict keys (remove 'module.' prefix from DistributedDataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Print model info
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create RectifiedFlow wrapper
    rf = RectifiedFlow(
        model, 
        class_dropout_prob=0.1,
        t_eps=5e-2,
        P_mean=-0.8,
        P_std=0.8,
    )
    
    print(f"Generating {args.num_samples} samples with {args.num_steps} steps...")
    
    with torch.no_grad():
        # Generate class labels
        if args.class_label is not None:
            # Sample specific class
            y = torch.full((args.num_samples,), args.class_label, dtype=torch.long, device=device)
        elif args.num_samples == 100 and dataset_config.num_classes >= 10:
            # Generate 10 samples per class for nice 10x10 grid
            y = torch.arange(10, device=device).repeat_interleave(10)
        elif args.num_samples < dataset_config.num_classes:
            # Randomly sample unique classes
            y = torch.randperm(dataset_config.num_classes, device=device)[:args.num_samples]
        else:
            # Generate random labels cycling through classes
            y = torch.arange(dataset_config.num_classes, device=device).repeat(
                args.num_samples // dataset_config.num_classes + 1
            )[:args.num_samples]
        
        # Generate samples
        if args.save_gif:
            # Return all intermediate steps
            all_samples = rf.sample(
                shape=(args.num_samples, dataset_config.in_channels, 
                       dataset_config.image_size, dataset_config.image_size),
                steps=args.num_steps,
                device=device,
                y=y,
                cfg_scale=args.cfg_scale,
                shift=args.shift,
                cfg_interval=args.cfg_interval,
                return_all=True,
            )
            samples = all_samples[-1]
        else:
            samples = rf.sample(
                shape=(args.num_samples, dataset_config.in_channels, 
                       dataset_config.image_size, dataset_config.image_size),
                steps=args.num_steps,
                device=device,
                y=y,
                cfg_scale=args.cfg_scale,
                shift=args.shift,
                cfg_interval=args.cfg_interval,
            )
        
        # Denormalize
        mean = torch.tensor(dataset_config.normalize_mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(dataset_config.normalize_std, device=device).view(1, -1, 1, 1)
        samples = samples * std + mean
        samples = samples.clamp(0, 1)
        
        # Determine grid layout
        if args.num_samples == 100:
            nrow = 10
        else:
            nrow = int(args.num_samples ** 0.5)
        
        # Save samples
        save_image(samples, output_path, nrow=nrow)
        print(f"Saved samples to {output_path}")

        # Save GIF if requested
        if args.save_gif:
            print(f"Creating GIF with {len(all_samples)} frames...")
            frames = []
            for i in range(len(all_samples)):
                # Denormalize and clamp
                frame = all_samples[i] * std + mean
                frame = frame.clamp(0, 1)
                
                # Create grid for this step
                grid = make_grid(frame, nrow=nrow)
                
                # Convert to PIL Image
                # Add 0.5 for rounding, then to uint8
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                frames.append(im)
            
            # Save GIF
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,  # 100ms per frame
                loop=0,
            )
            print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    sample()

#!/usr/bin/env python3
"""CompHairHead Demo: Reconstruct, animate, and transfer hairstyles.

Usage:
    python scripts/demo.py --image <path_to_portrait.jpg> [--output outputs/]
    python scripts/demo.py --mode animate --avatar outputs/avatar.pt
    python scripts/demo.py --mode transfer --source <src_avatar.pt> --target <tgt_avatar.pt>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="CompHairHead Demo")
    parser.add_argument("--mode", choices=["reconstruct", "animate", "transfer"],
                        default="reconstruct", help="Pipeline mode")
    parser.add_argument("--image", type=str, help="Input portrait image path")
    parser.add_argument("--avatar", type=str, help="Pre-built avatar path (.pt)")
    parser.add_argument("--source", type=str, help="Source avatar for transfer")
    parser.add_argument("--target", type=str, help="Target avatar for transfer")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/mps/cuda/cpu")
    args = parser.parse_args()

    # Load config
    from comp_hair_head.config import load_config
    config = load_config(args.config)

    if args.device != "auto":
        config.device.device = args.device

    if args.mode == "reconstruct":
        if not args.image:
            parser.error("--image required for reconstruct mode")
        run_reconstruct(args.image, args.output, config)

    elif args.mode == "animate":
        if not args.avatar:
            parser.error("--avatar required for animate mode")
        run_animate(args.avatar, args.output, config)

    elif args.mode == "transfer":
        if not args.source or not args.target:
            parser.error("--source and --target required for transfer mode")
        run_transfer(args.source, args.target, args.output, config)


def run_reconstruct(image_path: str, output_dir: str, config):
    """Run full reconstruction pipeline."""
    from comp_hair_head.pipeline.reconstruct import ReconstructionPipeline

    print(f"\n🎭 CompHairHead Reconstruction")
    print(f"  Input: {image_path}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {config.get_device()}\n")

    pipeline = ReconstructionPipeline(config)
    result = pipeline.reconstruct(image_path, output_dir)

    print(f"\n✅ Done!")
    print(f"  Hair Gaussians: {result['G_hair_local'].num_gaussians}")
    print(f"  Bald Gaussians: {result['G_bald_local'].num_gaussians}")
    print(f"  Saved to: {output_dir}/avatar.pt")


def run_animate(avatar_path: str, output_dir: str, config):
    """Run animation driving."""
    from comp_hair_head.pipeline.animate import AnimationPipeline

    print(f"\n🎬 CompHairHead Animation")
    print(f"  Avatar: {avatar_path}")
    print(f"  Output: {output_dir}\n")

    # Load avatar
    checkpoint = torch.load(avatar_path, map_location="cpu")
    print("  Avatar loaded. Animation pipeline setup...")

    # Create a simple head rotation sequence for demo
    T = 60
    pose_sequence = torch.zeros(T, 15)
    for t in range(T):
        # Head rotation: yaw back and forth
        angle = 0.3 * torch.sin(torch.tensor(2 * 3.14159 * t / T))
        pose_sequence[t, 1] = angle  # neck Y rotation

    print(f"  Generated {T}-frame demo sequence")
    print(f"  (Full animation requires loaded avatar state)")


def run_transfer(source_path: str, target_path: str, output_dir: str, config):
    """Run hairstyle transfer."""
    from comp_hair_head.pipeline.transfer import HairstyleTransfer

    print(f"\n💇 CompHairHead Hairstyle Transfer")
    print(f"  Source: {source_path}")
    print(f"  Target: {target_path}")
    print(f"  Output: {output_dir}\n")

    print("  (Transfer requires two pre-reconstructed avatars)")


if __name__ == "__main__":
    main()

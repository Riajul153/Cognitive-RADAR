"""Script to launch DRL agent training for adaptive beamforming."""

from __future__ import annotations

import argparse
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.trainer import BeamTrackingTrainer


def main():
    parser = argparse.ArgumentParser(description="Train DRL agent for adaptive beamforming radar tracking.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["SAC", "TD3"],
        help="DRL algorithm to train (overrides config).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Total training timesteps (overrides config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        help="PyTorch device for policy/critic networks (overrides config).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        help="Number of parallel training environments (overrides config).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .zip model file to resume training from.",
    )

    args = parser.parse_args()

    # Verify config path
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    print(f"Loading configuration from: {args.config}")
    trainer = BeamTrackingTrainer(args.config)

    # Overrides if provided in CLI
    if args.algo:
        trainer.config["training"]["algorithm"] = args.algo
    if args.steps:
        trainer.config["training"]["total_timesteps"] = args.steps
    if args.resume:
        trainer.config["training"]["resume_path"] = args.resume
    if args.device:
        trainer.config["training"]["device"] = args.device
    if args.n_envs:
        trainer.config["training"]["n_envs"] = args.n_envs

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully.")
        sys.exit(0)


if __name__ == "__main__":
    main()

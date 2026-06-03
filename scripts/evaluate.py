"""Evaluation script comparing the trained DRL agent with conjugate phase and fixed baselines."""

from __future__ import annotations

import argparse
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC, TD3
from src.environment.beam_tracking_env import BeamTrackingEnv
from src.antenna.steering import compute_optimal_phases, angular_distance


def evaluate_baselines(env: BeamTrackingEnv, model_path: str | None, num_episodes: int = 5) -> dict[str, dict[str, np.ndarray]]:
    """Evaluates the DRL model, Conjugate Phase Oracle, and Fixed Boresight baseline.

    Args:
        env: The evaluation environment.
        model_path: Path to the trained zip model. If None, only runs baselines.
        num_episodes: Number of episodes to run.

    Returns:
        Dictionary containing tracked metrics for each method.
    """
    # Load model
    agent = None
    if model_path:
        # Detect if model is SAC or TD3
        if "TD3" in model_path or "td3" in model_path:
            agent = TD3.load(model_path)
            print("Loaded TD3 model.")
        else:
            agent = SAC.load(model_path)
            print("Loaded SAC model.")

    methods = ["oracle", "fixed"]
    if agent:
        methods.insert(0, "drl")

    # Storage for metrics
    results = {
        method: {
            "gains": [],
            "errors_deg": [],
            "rewards": [],
        }
        for method in methods
    }

    # Fixed seed for fair comparison
    seed_base = 12345

    for method in methods:
        print(f"Evaluating method: {method.upper()}...")
        
        for ep in range(num_episodes):
            obs, info = env.reset(seed=seed_base + ep)
            env.reward_computer.reset()
            
            done = False
            ep_gains = []
            ep_errors_deg = []
            ep_rewards = []
            
            while not done:
                # Compute action based on method
                if method == "drl":
                    action, _ = agent.predict(obs, deterministic=True)
                    # Environment step takes action in [-1.0, 1.0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    ep_gains.append(info["normalized_gain"])
                    ep_errors_deg.append(info["tracking_error_deg"])
                    ep_rewards.append(reward)
                    
                elif method == "oracle":
                    # Oracle knows target coordinates perfectly and uses conjugate phase
                    t_theta, t_phi = env.target.get_angular_position()
                    opt_phases = compute_optimal_phases(env.array, t_theta, t_phi)
                    action = (opt_phases / np.pi).astype(np.float32)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    ep_gains.append(info["normalized_gain"])
                    ep_errors_deg.append(info["tracking_error_deg"])
                    ep_rewards.append(reward)
                    
                elif method == "fixed":
                    # Fixed boresight baseline (phases set to zero, points at boresight)
                    zero_phases = np.zeros(env.array.n_elements, dtype=np.float32)
                    action = zero_phases
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    ep_gains.append(info["normalized_gain"])
                    ep_errors_deg.append(info["tracking_error_deg"])
                    ep_rewards.append(reward)

            results[method]["gains"].append(ep_gains)
            results[method]["errors_deg"].append(ep_errors_deg)
            results[method]["rewards"].append(ep_rewards)

    # Convert lists to arrays
    for method in methods:
        results[method]["gains"] = np.array(results[method]["gains"])
        results[method]["errors_deg"] = np.array(results[method]["errors_deg"])
        results[method]["rewards"] = np.array(results[method]["rewards"])

    return results


def print_metrics(results: dict[str, dict[str, np.ndarray]]) -> None:
    """Computes and prints statistical comparison table."""
    print("\n" + "=" * 65)
    print(f"{'Method':<10} | {'Mean Gain (0-1)':<15} | {'Mean Error (deg)':<15} | {'Mean Step Reward':<15}")
    print("=" * 65)
    
    for method, data in results.items():
        mean_gain = np.mean(data["gains"])
        mean_error = np.mean(data["errors_deg"])
        mean_reward = np.mean(data["rewards"])
        print(f"{method.upper():<10} | {mean_gain:15.4f} | {mean_error:15.2f}° | {mean_reward:15.4f}")
    
    print("=" * 65)


def plot_results(results: dict[str, dict[str, np.ndarray]], save_path: str = "evaluation_results.png") -> None:
    """Generates comparison plots for gain and tracking error."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    colors = {"drl": "#00f0ff", "oracle": "#00ff88", "fixed": "#ff4466"}
    labels = {"drl": "DRL Agent", "oracle": "Conjugate Phase (Oracle)", "fixed": "Fixed Boresight"}

    # Average over episodes
    for method, data in results.items():
        # Shape is (num_episodes, steps)
        steps = np.arange(data["gains"].shape[1])
        
        mean_gains = np.mean(data["gains"], axis=0)
        std_gains = np.std(data["gains"], axis=0)
        
        mean_errors = np.mean(data["errors_deg"], axis=0)
        std_errors = np.std(data["errors_deg"], axis=0)

        # Plot Gain
        axes[0].plot(steps, mean_gains, color=colors[method], label=labels[method], linewidth=2)
        axes[0].fill_between(
            steps,
            np.clip(mean_gains - std_gains, 0, 1),
            np.clip(mean_gains + std_gains, 0, 1),
            color=colors[method],
            alpha=0.15,
        )

        # Plot Error
        axes[1].plot(steps, mean_errors, color=colors[method], label=labels[method], linewidth=2)
        axes[1].fill_between(
            steps,
            np.maximum(mean_errors - std_errors, 0),
            mean_errors + std_errors,
            color=colors[method],
            alpha=0.15,
        )

    # Styling
    axes[0].set_ylabel("Normalized Array Gain", fontsize=11)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title("Adaptive Beamforming Tracking Performance", fontsize=14, pad=15)
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].set_ylabel("Tracking Error (Degrees)", fontsize=11)
    axes[1].set_xlabel("Time Steps (10ms)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_ylim(-2.0, 45.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nEvaluation plots saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained beamforming agent.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/SAC_final_model.zip",
        help="Path to trained agent zip file. Set to 'none' to run only baselines.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="evaluation_results.png",
        help="Path to output plot image.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config not found at {args.config}")
        sys.exit(1)

    model_path = args.model
    if model_path.lower() == "none" or not os.path.exists(model_path):
        print(f"Warning: Model not found at '{model_path}'. Running baseline comparisons only.")
        model_path = None

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize environment
    print("Initializing environment...")
    env = BeamTrackingEnv(
        n_rows=config["antenna"]["n_rows"],
        n_cols=config["antenna"]["n_cols"],
        frequency_hz=config["antenna"]["frequency_hz"],
        element_spacing=config["antenna"]["element_spacing"],
        target_config=config["target"],
        reward_config=config["environment"]["reward"],
        success_config=config["environment"].get("success"),
        dt=config["environment"]["dt"],
        episode_length=config["environment"]["episode_length"],
        rng_seed=999,
    )

    # Run evaluations
    results = evaluate_baselines(env, model_path, num_episodes=args.episodes)
    
    # Print metrics
    print_metrics(results)
    
    # Save plots
    plot_results(results, args.out)

    env.close()


if __name__ == "__main__":
    main()

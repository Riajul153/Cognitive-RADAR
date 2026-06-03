"""Script to evaluate all saved checkpoints and find the best performing one."""

import os
import glob
import argparse
import numpy as np
import yaml
from stable_baselines3 import SAC

from src.environment.beam_tracking_env import BeamTrackingEnv

def evaluate_model(model_path, env_fn, num_episodes, seeds):
    """Evaluates a single model over a set of episodes."""
    print(f"Evaluating {os.path.basename(model_path)}...")
    agent = SAC.load(model_path, device="cpu")
    env = env_fn()
    
    gains = []
    locked_fractions = []
    
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        done = False
        ep_gains = []
        locked_steps = 0
        ep_steps = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_gains.append(info.get("normalized_gain", 0.0))
            if info.get("tracking_locked", False):
                locked_steps += 1
            ep_steps += 1
            
        gains.append(np.mean(ep_gains))
        locked_fractions.append(locked_steps / max(ep_steps, 1))
        
    env.close()
    
    return {
        "mean_gain": np.mean(gains),
        "mean_locked_fraction": np.mean(locked_fractions),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/v100_optimized.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default="logs/checkpoints/")
    parser.add_argument("--episodes", type=int, default=10) # Lower for speed, just finding top candidates
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    env_cfg = config["environment"]
    
    def make_env():
        return BeamTrackingEnv(
            n_rows=env_cfg.get("n_rows", 8),
            n_cols=env_cfg.get("n_cols", 8),
            frequency_hz=env_cfg.get("frequency_hz", 10e9),
            element_spacing=env_cfg.get("element_spacing", 0.5),
            target_config=env_cfg.get("target", {}),
            reward_config=env_cfg.get("reward", {}),
            success_config=env_cfg.get("success", {}),
            dt=env_cfg.get("dt", 0.01),
            episode_length=env_cfg.get("episode_length", 500),
            action_mode=env_cfg.get("action_mode", "parametric"),
            parametric_type=env_cfg.get("parametric_type", "incremental"),
            max_angular_step_deg=env_cfg.get("max_angular_step_deg", 2.0),
        )

    checkpoints = glob.glob("logs/checkpoints/SAC_beam_tracking_*.zip")
    if not checkpoints:
        print(f"No checkpoints found in logs/checkpoints/")
        return
        
    # Also evaluate the best_model and final model for comparison
    checkpoints.append("models/best_by_gain/best_model.zip")
    checkpoints.append("models/SAC_final_model.zip")
    
    seeds = [1000 + i for i in range(args.episodes)]
    
    results = []
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            try:
                metrics = evaluate_model(ckpt, make_env, args.episodes, seeds)
                results.append((ckpt, metrics["mean_gain"], metrics["mean_locked_fraction"]))
            except Exception as e:
                print(f"Failed to evaluate {ckpt}: {e}")
                
    # Sort by mean gain descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*70)
    print(f"{'Checkpoint':<45} | {'Mean Gain':<10} | {'Locked %':<10}")
    print("-" * 70)
    for ckpt, gain, lf in results:
        name = os.path.basename(ckpt)
        print(f"{name:<45} | {gain:<10.4f} | {lf:<10.1%}")
    print("="*70)

if __name__ == "__main__":
    main()

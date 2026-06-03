"""Script to measure initial lock-in time for SAC vs EKF."""

import argparse
import numpy as np
import yaml
from stable_baselines3 import SAC

from src.environment.beam_tracking_env import BeamTrackingEnv
from src.baselines.monopulse_tracker import MonopulseTracker

def measure_lock_time(agent, env_fn, num_episodes, seeds, is_baseline=False):
    """Measures the number of steps to achieve first lock from an offset."""
    env = env_fn()
    lock_times = []
    
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        
        # --- Force initial tracking error ---
        # The target is at info["target_angles"]
        t_theta, t_phi = info["target_angles"]
        
        # Offset the beam by 20 degrees in theta and 20 degrees in phi
        offset_rad = np.radians(20.0)
        env.unwrapped.beam_theta_cmd = np.clip(t_theta + offset_rad, 0.0, env.unwrapped.elevation_max_rad)
        env.unwrapped.beam_phi_cmd = t_phi + offset_rad
        
        # Recompute initial observation so the agent sees the error
        obs = env.unwrapped._get_obs()
        
        if is_baseline:
            agent.ekf = None
            
        done = False
        steps = 0
        locked = False
        
        while not done:
            if is_baseline:
                action, _ = agent.predict(obs, info=info)
            else:
                action, _ = agent.predict(obs, deterministic=True)
                
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if info.get("tracking_locked", False):
                lock_times.append(steps)
                locked = True
                break
                
            done = terminated or truncated
            
        if not locked:
            # If it never locked during the episode (500 steps)
            lock_times.append(500)
            
    env.close()
    
    return lock_times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/v100_optimized.yaml")
    parser.add_argument("--model", type=str, default="logs/checkpoints/SAC_beam_tracking_2950000_steps.zip")
    parser.add_argument("--episodes", type=int, default=50)
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
            dt=env_cfg.get("dt", 0.01), # 10ms
            episode_length=env_cfg.get("episode_length", 500),
            action_mode=env_cfg.get("action_mode", "parametric"),
            parametric_type=env_cfg.get("parametric_type", "incremental"),
            max_angular_step_deg=env_cfg.get("max_angular_step_deg", 2.0),
        )

    print(f"Loading RL Model from {args.model}...")
    rl_agent = SAC.load(args.model, device="cpu")
    
    print("Initializing EKF Monopulse Baseline...")
    target_cfg = env_cfg.get("target", {})
    baseline_agent = MonopulseTracker(
        dt=env_cfg.get("dt", 0.01),
        max_angular_step_deg=env_cfg.get("max_angular_step_deg", 2.0),
        noise_std_theta_deg=0.5,
        noise_std_phi_deg=0.5,
        tau=target_cfg.get("correlation_time", 2.0),
        sigma_a=target_cfg.get("acceleration_std", 30.0),
    )
    
    seeds = [2000 + i for i in range(args.episodes)]
    
    print(f"\nMeasuring EKF Lock-in Time over {args.episodes} episodes...")
    ekf_times = measure_lock_time(baseline_agent, make_env, args.episodes, seeds, is_baseline=True)
    
    print(f"Measuring SAC Lock-in Time over {args.episodes} episodes...")
    sac_times = measure_lock_time(rl_agent, make_env, args.episodes, seeds, is_baseline=False)
    
    dt = env_cfg.get("dt", 0.01)
    
    ekf_mean_steps = np.mean(ekf_times)
    sac_mean_steps = np.mean(sac_times)
    
    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'EKF Baseline':<10} | {'SAC RL Agent':<10}")
    print("-" * 50)
    print(f"{'Mean Lock-in Steps':<25} | {ekf_mean_steps:<10.1f} | {sac_mean_steps:<10.1f}")
    print(f"{'Mean Lock-in Time (ms)':<25} | {ekf_mean_steps * dt * 1000:<10.1f} | {sac_mean_steps * dt * 1000:<10.1f}")
    print("="*50)

if __name__ == "__main__":
    main()

"""Benchmarking script to compare SAC RL agent against traditional EKF Monopulse baseline."""

import argparse
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.beam_tracking_env import BeamTrackingEnv
from src.baselines.monopulse_tracker import MonopulseTracker

def evaluate_agent(agent, env_fn, num_episodes, seeds, is_baseline=False):
    """Evaluates an agent over a set of episodes with specific seeds."""
    env = env_fn()
    
    gains = []
    errors = []
    successes = []
    locked_fractions = []
    
    for ep, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        if is_baseline:
            agent.ekf = None # Reset EKF state for new episode
            
        done = False
        ep_gains = []
        ep_errors = []
        locked_steps = 0
        ep_steps = 0
        
        while not done:
            if is_baseline:
                action, _ = agent.predict(obs, info=info)
            else:
                action, _ = agent.predict(obs, deterministic=True)
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ep_gains.append(info.get("normalized_gain", 0.0))
            ep_errors.append(info.get("tracking_error_deg", 90.0))
            if info.get("tracking_locked", False):
                locked_steps += 1
            ep_steps += 1
            
        gains.append(np.mean(ep_gains))
        errors.append(np.mean(ep_errors))
        lf = locked_steps / max(ep_steps, 1)
        locked_fractions.append(lf)
        successes.append(1.0 if lf >= 0.8 else 0.0) # Assume 80% lock is success
        
    env.close()
    
    return {
        "mean_gain": np.mean(gains),
        "mean_error": np.mean(errors),
        "mean_locked_fraction": np.mean(locked_fractions),
        "success_rate": np.mean(successes),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/v100_optimized.yaml")
    parser.add_argument("--model", type=str, default="models/best_by_gain/best_model.zip")
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
            dt=env_cfg.get("dt", 0.01),
            episode_length=env_cfg.get("episode_length", 500),
            action_mode=env_cfg.get("action_mode", "parametric"),
            parametric_type=env_cfg.get("parametric_type", "incremental"),
            max_angular_step_deg=env_cfg.get("max_angular_step_deg", 2.0),
        )

    print("Evaluating EKF Monopulse Baseline...")
    target_cfg = env_cfg.get("target", {})
    baseline_agent = MonopulseTracker(
        dt=env_cfg.get("dt", 0.01),
        max_angular_step_deg=env_cfg.get("max_angular_step_deg", 2.0),
        noise_std_theta_deg=0.5,
        noise_std_phi_deg=0.5,
        tau=target_cfg.get("correlation_time", 2.0),
        sigma_a=target_cfg.get("acceleration_std", 30.0),
    )
    seeds = [1000 + i for i in range(args.episodes)]
    baseline_metrics = evaluate_agent(baseline_agent, make_env, args.episodes, seeds, is_baseline=True)
    
    print("Evaluating SAC RL Agent...")
    rl_agent = SAC.load(args.model, device="cpu")
    rl_metrics = evaluate_agent(rl_agent, make_env, args.episodes, seeds, is_baseline=False)
    
    print("Evaluating Codebook DQN Agent...")
    from stable_baselines3 import DQN
    dqn_model_path = "models/dqn/best_by_gain/best_model.zip"
    try:
        dqn_agent = DQN.load(dqn_model_path, device="cpu")
        def make_dqn_env():
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
                action_mode="codebook",
                elevation_max_deg=60.0,
                azimuth_max_deg=60.0,
            )
        dqn_metrics = evaluate_agent(dqn_agent, make_dqn_env, args.episodes, seeds, is_baseline=False)
    except Exception as e:
        print(f"Failed to load DQN model: {e}")
        dqn_metrics = {"mean_gain": 0.0, "mean_error": 0.0, "mean_locked_fraction": 0.0, "success_rate": 0.0}
    
    print("Evaluating 64D Raw SAC Agent...")
    raw_model_path = "models/sac_raw_64d/best_by_gain/best_model.zip"
    try:
        raw_agent = SAC.load(raw_model_path, device="cpu")
        def make_raw_env():
            return BeamTrackingEnv(
                n_rows=env_cfg.get("n_rows", 8),
                n_cols=env_cfg.get("n_cols", 8),
                action_mode="raw"
            )
        raw_metrics = evaluate_agent(raw_agent, make_raw_env, args.episodes, seeds, is_baseline=False)
    except Exception as e:
        print(f"Failed to load 64D Raw SAC model: {e}")
        raw_metrics = {"mean_gain": 0.0, "mean_error": 0.0, "mean_locked_fraction": 0.0, "success_rate": 0.0}
    
    print("\n" + "="*90)
    print(f"{'Metric':<25} | {'EKF Baseline':<12} | {'Codebook DQN':<12} | {'64D Raw SAC':<12} | {'Parametric SAC':<12}")
    print("-" * 90)
    print(f"{'Mean Gain':<25} | {baseline_metrics['mean_gain']:<12.4f} | {dqn_metrics['mean_gain']:<12.4f} | {raw_metrics['mean_gain']:<12.4f} | {rl_metrics['mean_gain']:<12.4f}")
    print(f"{'Mean Error (deg)':<25} | {baseline_metrics['mean_error']:<12.2f} | {dqn_metrics['mean_error']:<12.2f} | {raw_metrics['mean_error']:<12.2f} | {rl_metrics['mean_error']:<12.2f}")
    print(f"{'Locked Fraction':<25} | {baseline_metrics['mean_locked_fraction']:<12.1%} | {dqn_metrics['mean_locked_fraction']:<12.1%} | {raw_metrics['mean_locked_fraction']:<12.1%} | {rl_metrics['mean_locked_fraction']:<12.1%}")
    print(f"{'Success Rate':<25} | {baseline_metrics['success_rate']:<12.1%} | {dqn_metrics['success_rate']:<12.1%} | {raw_metrics['success_rate']:<12.1%} | {rl_metrics['success_rate']:<12.1%}")
    print("="*90)

if __name__ == "__main__":
    main()

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.environment.beam_tracking_env import BeamTrackingEnv
from src.agents.jepa_agent import JEPABeamTrackingAgent
from src.models.jepa import JEPAConfig
from src.models.cost_module import CostConfig
from src.models.planner import PlannerConfig

def evaluate_agent(env, agent, n_episodes=5):
    """Evaluates the agent for n_episodes and returns average metrics."""
    total_gain = 0.0
    total_error = 0.0
    total_successes = 0
    total_steps = 0
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        agent.reset_planner()
        done = False
        truncated = False
        
        ep_gain = 0.0
        ep_error = 0.0
        ep_steps = 0
        
        while not (done or truncated):
            action = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            obs = next_obs
            
            ep_gain += info["normalized_gain"]
            ep_error += info["tracking_error_deg"]
            ep_steps += 1
            
        total_gain += ep_gain / ep_steps
        total_error += ep_error / ep_steps
        total_steps += 1
        if info.get("is_success", False):
            total_successes += 1
            
    metrics = {
        "mean_gain": total_gain / n_episodes,
        "mean_error": total_error / n_episodes,
        "success_rate": total_successes / n_episodes
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="jepa_run_1")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--initial_collect", type=int, default=20000)
    parser.add_argument("--collect_per_iter", type=int, default=5000)
    parser.add_argument("--epochs_per_iter", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    
    print(f"=== Starting JEPA Training: {args.run_name} ===")
    
    # 1. Initialize Environment
    # We use crossed dipole settings to avoid nulls, ensuring physics are learnable.
    # We will use parametric incremental mode.
    env = BeamTrackingEnv(
        n_rows=8,
        n_cols=8,
        action_mode="parametric",
        parametric_type="incremental",
        max_angular_step_deg=2.0
    )
    
    # 2. Initialize Agent
    jepa_config = JEPAConfig(obs_dim=10, action_dim=2, latent_dim=32, vicreg_lambda=25.0, vicreg_mu=25.0, vicreg_nu=1.0)
    cost_config = CostConfig(latent_dim=32)
    planner_config = PlannerConfig(horizon=15, population=256, elite_fraction=0.1, n_iterations=3)
    
    agent = JEPABeamTrackingAgent(jepa_config, cost_config, planner_config, device=args.device)
    
    # 3. Phase A: Initial Data Collection (Random/Noisy Exploration)
    print(f"Collecting {args.initial_collect} initial transitions...")
    obs, info = env.reset()
    for _ in range(args.initial_collect):
        # We need coverage of the state-action space.
        # Random action
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Add to dataset (cost module targets 1 - received_power, so we pass received_power)
        # Power is info["target_gain"]
        agent.dataset.add(obs, action, next_obs, info["normalized_gain"])
        
        if done or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs
            
    print(f"Initial collection complete. Dataset size: {len(agent.dataset)}")
    
    # 4. Main Training Loop
    best_success = -1.0
    
    for it in range(args.iterations):
        print(f"\n--- Iteration {it+1}/{args.iterations} ---")
        
        # Train Offline
        print(f"Training for {args.epochs_per_iter} epochs...")
        hist = agent.train_offline(
            n_epochs=args.epochs_per_iter,
            batch_size=args.batch_size,
            lr=args.lr,
            writer=writer,
            log_freq=50
        )
        
        # Evaluate
        print("Evaluating agent planner...")
        metrics = evaluate_agent(env, agent, n_episodes=5)
        print(f"Eval: Success Rate = {metrics['success_rate']*100:.1f}% | Mean Gain = {metrics['mean_gain']:.3f} | Mean Error = {metrics['mean_error']:.2f} deg")
        
        writer.add_scalar("eval/success_rate", metrics['success_rate'], it)
        writer.add_scalar("eval/mean_gain", metrics['mean_gain'], it)
        writer.add_scalar("eval/mean_error", metrics['mean_error'], it)
        
        # Save Best
        if metrics['success_rate'] >= best_success:
            best_success = metrics['success_rate']
            save_path = os.path.join(run_dir, "best_model.pt")
            agent.save_checkpoint(save_path, metadata=metrics)
            print(f"[*] New best model saved! ({best_success*100:.1f}%)")
            
        # Collect More Data (On-Policy with exploration)
        if it < args.iterations - 1:
            print(f"Collecting {args.collect_per_iter} new transitions with planner...")
            obs, info = env.reset()
            agent.reset_planner()
            for _ in range(args.collect_per_iter):
                # Epsilon-greedy exploration
                if np.random.rand() < 0.15:
                    action = env.action_space.sample()
                else:
                    action = agent.act(obs)
                    # Add small exploration noise
                    action = np.clip(action + np.random.normal(0, 0.1, size=2), -1.0, 1.0)
                    
                next_obs, reward, done, truncated, info = env.step(action)
                agent.dataset.add(obs, action, next_obs, info["normalized_gain"])
                
                if done or truncated:
                    obs, info = env.reset()
                    agent.reset_planner()
                else:
                    obs = next_obs
                    
    print("\nTraining Complete.")
    writer.close()

if __name__ == "__main__":
    main()

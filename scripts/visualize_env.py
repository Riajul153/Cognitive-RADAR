"""Run BeamTrackingEnv with oracle/random/fixed actions and stream to the web dashboard."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.antenna.steering import compute_optimal_phases
from src.environment.beam_tracking_env import BeamTrackingEnv
from src.utils.dashboard_client import DashboardClient


def build_beam_pattern_grid(beamformer, theta_steps: int = 15, phi_steps: int = 30) -> dict:
    """Coarse (theta, phi) grid for 3D dashboard surface plot."""
    theta_grid = np.linspace(0.0, np.pi / 2.0, theta_steps)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, phi_steps)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    power_grid = beamformer.compute_beam_pattern(theta_mesh, phi_mesh)
    return {"theta": theta_grid, "phi": phi_grid, "power": power_grid}


def select_action(env: BeamTrackingEnv, policy: str, rng: np.random.Generator) -> np.ndarray:
    """Pick an action vector for the current env state."""
    if policy == "oracle":
        t_theta, t_phi = env.target.get_angular_position()
        phases = compute_optimal_phases(env.array, t_theta, t_phi)
        return (phases / np.pi).astype(np.float32)
    if policy == "fixed":
        return np.zeros(env.action_space.shape, dtype=np.float32)
    return rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)


def stream_frame(
    client: DashboardClient,
    env: BeamTrackingEnv,
    episode: int,
    reward: float,
    label: str,
    beam_pattern: dict | None,
) -> None:
    """Push one dashboard update from the current env state."""
    t_theta, t_phi = env.target.get_angular_position()
    b_theta, b_phi = env.beamformer.estimate_beam_direction()
    client.send_state(
        step=env.current_step,
        episode=episode,
        algorithm=label,
        target_pos=env.target.get_position(),
        target_angles=(t_theta, t_phi),
        beam_angles=(b_theta, b_phi),
        gain=env.last_gain,
        error_deg=float(np.degrees(env.last_error)),
        reward=float(reward),
        phases=env.beamformer.get_current_weights(),
        beam_pattern=beam_pattern,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize BeamTrackingEnv on the web dashboard (no RL training required)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["oracle", "random", "fixed"],
        default="oracle",
        help="Control policy: conjugate-phase oracle, random phases, or fixed boresight.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (loops forever if 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for env reset and random policy.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=50,
        help="Milliseconds to sleep between steps (controls animation speed).",
    )
    parser.add_argument(
        "--stream-freq",
        type=int,
        default=1,
        help="Send a dashboard frame every N environment steps.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run in terminal only (env.render), do not use WebSocket.",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Also print env.render() lines to the console.",
    )
    parser.add_argument(
        "--ws-host",
        type=str,
        default="localhost",
        help="WebSocket broker host (must match serve_dashboard.py).",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket broker port.",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="Dashboard HTTP port (printed in instructions; start serve_dashboard.py).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config not found at {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ant = config["antenna"]
    env = BeamTrackingEnv(
        n_rows=ant["n_rows"],
        n_cols=ant["n_cols"],
        frequency_hz=ant["frequency_hz"],
        element_spacing=ant["element_spacing"],
        target_config=config["target"],
        reward_config=config["environment"]["reward"],
        success_config=config["environment"].get("success"),
        dt=config["environment"]["dt"],
        episode_length=config["environment"]["episode_length"],
        rng_seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    policy_labels = {
        "oracle": "ORACLE (conj. phase)",
        "random": "RANDOM phases",
        "fixed": "FIXED boresight",
    }
    label = policy_labels[args.policy]

    client = None
    if not args.no_dashboard:
        dash = config.get("dashboard", {})
        client = DashboardClient(
            host=args.ws_host,
            port=args.ws_port or dash.get("websocket_port", 8765),
            n_rows=ant["n_rows"],
            n_cols=ant["n_cols"],
        )
        client.start()
        dashboard_url = f"http://{args.ws_host}:{args.http_port}/"
        print()
        print("=" * 60)
        print("  DASHBOARD — open this in your browser:")
        print(f"  {dashboard_url}")
        print()
        print("  Start the broker first (separate terminal):")
        print("  python scripts/serve_dashboard.py")
        print()
        print("  Click DEMO MODE off if it is still animating demo data.")
        print("=" * 60)
        print()

    beam_pattern = build_beam_pattern_grid(env.beamformer)
    episode = 0
    delay_s = max(0.0, args.delay_ms / 1000.0)

    try:
        while True:
            episode += 1
            obs, info = env.reset(seed=args.seed + episode)
            env.reward_computer.reset()

            if client is not None:
                stream_frame(client, env, episode, reward=0.0, label=label, beam_pattern=beam_pattern)

            if args.terminal:
                env.render()

            done = False
            while not done:
                action = select_action(env, args.policy, rng)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if client is not None and env.current_step % args.stream_freq == 0:
                    stream_frame(client, env, episode, reward, label, beam_pattern)

                if args.terminal:
                    env.render()

                if delay_s > 0:
                    time.sleep(delay_s)

            if args.episodes > 0 and episode >= args.episodes:
                break

    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        if client is not None:
            time.sleep(0.5)
            client.stop()
        env.close()


if __name__ == "__main__":
    main()

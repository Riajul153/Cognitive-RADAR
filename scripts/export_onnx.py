import os
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import SAC, TD3

def export_sb3_policy(model_path, out_path, obs_dim=10):
    """Exports the deterministic policy network of an SB3 SAC/TD3 model to ONNX."""
    print(f"Loading SB3 model from {model_path}...")
    
    # Try loading as SAC, then TD3
    try:
        model = SAC.load(model_path)
        print("Loaded as SAC model.")
    except Exception:
        model = TD3.load(model_path)
        print("Loaded as TD3 model.")
        
    policy = model.policy
    policy.eval()

    # Create a dummy observation matching the environment obs_space
    dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32, device=policy.device)
    
    class OnnxPolicyWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            
        def forward(self, obs):
            # Deterministic action
            return self.policy(obs, deterministic=True)

    wrapped_policy = OnnxPolicyWrapper(policy)

    print(f"Exporting to ONNX at {out_path}...")
    torch.onnx.export(
        wrapped_policy,
        dummy_obs,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
    )
    print("Export successful!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .zip model file")
    parser.add_argument("--out_path", type=str, default="model.onnx", help="Path to save the .onnx file")
    parser.add_argument("--obs_dim", type=int, default=10, help="Observation dimension")
    args = parser.parse_args()
    
    export_sb3_policy(args.model_path, args.out_path, obs_dim=args.obs_dim)

if __name__ == "__main__":
    main()

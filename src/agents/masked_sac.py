import torch as th
from torch.nn import functional as F
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

class MaskedSAC(SAC):
    """Custom SAC algorithm with Outage Filtering (Critic Loss Masking).
    
    This injects radar domain knowledge into the model-free RL algorithm.
    By monitoring the normalized `received_power` in the observation, we can detect
    when the target is traversing a physical hardware null. During these outages,
    the monopulse error signal is pure noise. 
    
    We apply a binary mask to the Critic's TD-error during these frames to prevent
    catastrophic forgetting and representation collapse.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        masked_fractions = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            
            # --- OUTAGE FILTER MASKING ---
            # Extract power from the observations.
            # Observation is stacked, e.g. [batch_size, stack_size * obs_dim]
            # Since obs_dim is 10, the most recent frame is the last 10 elements.
            # `received_power` is at index 2 of that frame. So index is -10 + 2 = -8
            power_norm = replay_data.observations[:, -8]
            power = (power_norm + 1.0) / 2.0  # Denormalize to [0, 1]
            
            # Create boolean mask. 1.0 if power is valid, 0.0 if in a null.
            valid_mask = (power > 0.15).float().unsqueeze(1)
            
            # Track how many transitions were masked out this batch
            masked_frac = 1.0 - valid_mask.mean().item()
            masked_fractions.append(masked_frac)
            # -----------------------------

            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute masked critic loss
            # Instead of standard MSE, we do reduction='none', multiply by mask, and take mean.
            critic_loss = 0.5 * sum(
                (F.mse_loss(current_q, target_q_values, reduction='none') * valid_mask).mean() 
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute masked actor loss
            # We don't necessarily need to mask the actor loss because min_qf_pi is frozen
            # but masking it ensures we only optimize the policy on valid states.
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            
            actor_loss = ((ent_coef * log_prob - min_qf_pi) * valid_mask).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/masked_fraction", np.mean(masked_fractions))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

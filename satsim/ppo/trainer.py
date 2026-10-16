"""PPO trainer for satellite routing using a fused GAT+LSTM observation space.

Implements the core Proximal Policy Optimisation (PPO-Clip) algorithm over the
``SatelliteRouting-v0`` Gymnasium environment wrapped with
:class:`~satsim.ppo.obs_wrapper.FusedEmbeddingWrapper`.

Algorithm summary
-----------------
1. Collect ``rollout_steps`` transitions in the environment using the current policy.
2. Compute GAE returns and advantages from the rollout buffer.
3. For ``epochs_per_update`` epochs, split the batch into mini-batches and apply:
   - Clipped surrogate policy loss (PPO-Clip).
   - Value function MSE loss.
   - Entropy bonus.
4. Repeat until ``total_timesteps`` environment interactions are reached.

Public API
----------
- ``PPOTrainer`` — main training class.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from satsim.logging import get_logger
from satsim.ppo.actor_critic import PPOActorCritic
from satsim.ppo.config import PPOConfig
from satsim.ppo.obs_wrapper import FusedEmbeddingWrapper
from satsim.ppo.rollout_buffer import RolloutBuffer, Transition
from satsim.envs.routing_env import SatelliteRoutingEnv

logger = get_logger(__name__)


class PPOTrainer:
    """Trains a :class:`~satsim.ppo.actor_critic.PPOActorCritic` agent on the
    ``SatelliteRouting-v0`` environment with a 256-dim fused GAT+LSTM observation.

    Args:
        config: :class:`~satsim.ppo.config.PPOConfig` containing all
                hyperparameters and path configuration.
    """

    def __init__(self, config: PPOConfig) -> None:
        self.cfg = config
        self.device = torch.device(config.device)
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Save exact config used (reproducibility requirement) ----
        config.to_yaml(self._output_dir / "config_used.yaml")

        # ---- Build environment ----
        base_env = SatelliteRoutingEnv(max_hops=config.max_hops)
        self.env = FusedEmbeddingWrapper(
            env=base_env,
            gat_checkpoint=config.gat_checkpoint,
            gat_scaler=config.gat_scaler,
            lstm_checkpoint=config.lstm_checkpoint,
            lstm_scaler=config.lstm_scaler,
            device=config.device,
            obs_dim=config.obs_dim,
            use_action_mask=config.use_action_mask,
            use_progress_reward=config.use_progress_reward,
            include_dst_embedding=config.include_dst_embedding,
            prevent_loops=getattr(config, "prevent_loops", True),
        )

        # ---- Seed ----
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # ---- Actor-Critic network ----
        self.policy = PPOActorCritic(
            obs_dim=config.obs_dim,
            num_actions=config.num_actions,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
        ).to(self.device)

        # ---- Optimiser ----
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr, eps=1e-5)

        # ---- Rollout buffer ----
        self.buffer = RolloutBuffer(
            capacity=config.rollout_steps,
            obs_dim=config.obs_dim,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=self.device,
        )

        # ---- Counters ----
        self._global_step: int = 0
        self._update_count: int = 0

        # ---- Episode tracking (for logging) ----
        self._ep_returns: list[float] = []
        self._ep_lengths: list[int] = []
        self._ep_deliveries: list[float] = []
        self._current_ep_return: float = 0.0
        self._current_ep_length: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full PPO training loop until ``total_timesteps`` is reached.

        Returns:
            None
        """
        obs, _ = self.env.reset(seed=self.cfg.seed)
        start_time = time.time()

        logger.info(
            "PPO training started",
            total_timesteps=self.cfg.total_timesteps,
            rollout_steps=self.cfg.rollout_steps,
            device=str(self.device),
        )

        try:
            while self._global_step < self.cfg.total_timesteps:
                # ---- Collect rollout ----
                obs = self._collect_rollout(obs)

                # ---- Compute returns and advantages ----
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    _, last_value = self.policy(obs_t)
                    last_value_scalar = last_value.squeeze().item()

                self.buffer.compute_returns_and_advantages(last_value=last_value_scalar)

                # ---- Learning rate annealing ----
                if getattr(self.cfg, "lr_annealing", True):
                    frac = max(0.0, 1.0 - (self._global_step / max(1, self.cfg.total_timesteps)))
                    current_lr = max(1e-5, self.cfg.lr * frac)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = current_lr

                # ---- Policy gradient update ----
                update_stats = self._ppo_update()

                self._update_count += 1
                self.buffer.clear()

                # ---- Logging ----
                if self._update_count % self.cfg.log_interval == 0:
                    elapsed = time.time() - start_time
                    fps = self._global_step / max(elapsed, 1e-6)
                    mean_ep_return = (
                        float(np.mean(self._ep_returns[-100:])) if self._ep_returns else float("nan")
                    )
                    mean_ep_length = (
                        float(np.mean(self._ep_lengths[-100:])) if self._ep_lengths else float("nan")
                    )
                    delivery_rate = (
                        float(np.mean(self._ep_deliveries[-100:])) if self._ep_deliveries else 0.0
                    )
                    logger.info(
                        "PPO update",
                        update=self._update_count,
                        global_step=self._global_step,
                        fps=round(fps, 1),
                        delivery_rate=round(delivery_rate, 4),
                        mean_ep_return=round(mean_ep_return, 4),
                        mean_ep_length=round(mean_ep_length, 2),
                        **{k: round(float(v), 6) for k, v in update_stats.items()},
                    )

                # ---- Periodic & latest checkpointing ----
                if self._update_count % self.cfg.save_interval == 0:
                    self._save_checkpoint(tag=f"update_{self._update_count:06d}")
                    self._save_checkpoint(tag="latest")

            # Final checkpoint
            self._save_checkpoint(tag="final")
            self._save_checkpoint(tag="latest")
            logger.info(
                "PPO training complete",
                global_step=self._global_step,
                updates=self._update_count,
            )
        except KeyboardInterrupt:
            pause_ckpt = self._save_checkpoint(tag="latest")
            logger.warning(
                "PPO training paused by user",
                global_step=self._global_step,
                updates=self._update_count,
                checkpoint=str(pause_ckpt),
            )
            print(f"\n[PAUSED] Training paused at step {self._global_step:,d} / {self.cfg.total_timesteps:,d}.")
            print(f"         Latest checkpoint saved to: {pause_ckpt}")
            print(f"         Resume anytime by adding: --resume {pause_ckpt}")

    def evaluate(self, num_episodes: int = 20) -> Dict[str, float]:
        """Run the greedy (deterministic) policy for ``num_episodes`` episodes.

        Args:
            num_episodes: Number of evaluation episodes to run.

        Returns:
            Dict with keys ``delivery_rate``, ``mean_return``, ``std_return``, ``mean_length``.
        """
        ep_returns = []
        ep_lengths = []
        deliveries = []

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = None
                if self.cfg.use_action_mask and hasattr(self.env, "get_action_mask"):
                    mask_np = self.env.get_action_mask()
                    mask_t = torch.tensor(mask_np, dtype=torch.bool, device=self.device).unsqueeze(0)

                action, _, _ = self.policy.act(obs_t, action_mask=mask_t, deterministic=True)
                obs, reward, terminated, truncated, step_info = self.env.step(action.item())
                ep_ret += reward
                ep_len += 1
                done = terminated or truncated

            is_delivered = bool(step_info.get("reward_components", {}).get("delivery", 0.0) > 0.0)
            deliveries.append(1.0 if is_delivered else 0.0)
            ep_returns.append(ep_ret)
            ep_lengths.append(ep_len)

        stats = {
            "delivery_rate": float(np.mean(deliveries)),
            "mean_return": float(np.mean(ep_returns)),
            "std_return": float(np.std(ep_returns)),
            "mean_length": float(np.mean(ep_lengths)),
        }
        logger.info("Evaluation complete", num_episodes=num_episodes, **stats)
        return stats

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load a previously saved PPO checkpoint.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.

        Returns:
            None
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"PPO checkpoint not found: '{checkpoint_path}'. "
                "Specify a valid path to a saved checkpoint file."
            )
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._global_step = ckpt.get("global_step", 0)
        self._update_count = ckpt.get("update_count", 0)
        self._ep_returns = ckpt.get("ep_returns", [])
        self._ep_lengths = ckpt.get("ep_lengths", [])
        self._ep_deliveries = ckpt.get("ep_deliveries", [])
        logger.info(
            "Checkpoint loaded",
            path=str(checkpoint_path),
            global_step=self._global_step,
            update_count=self._update_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_rollout(self, obs: np.ndarray) -> np.ndarray:
        """Collect ``rollout_steps`` environment transitions into the buffer.

        Args:
            obs: Current observation to start the rollout from.

        Returns:
            The observation at the end of the rollout (for bootstrapping).
        """
        self.policy.eval()

        for _ in range(self.cfg.rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            mask_t = None
            mask_np = None
            if self.cfg.use_action_mask and hasattr(self.env, "get_action_mask"):
                mask_np = self.env.get_action_mask()
                mask_t = torch.tensor(mask_np, dtype=torch.bool, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = self.policy.act(obs_t, action_mask=mask_t, deterministic=False)

            action_int = action.item()
            log_prob_float = log_prob.item()
            value_float = value.squeeze().item()

            next_obs, reward, terminated, truncated, info = self.env.step(action_int)
            done = terminated or truncated

            self.buffer.add(
                Transition(
                    obs=obs,
                    action=action_int,
                    log_prob=log_prob_float,
                    value=value_float,
                    reward=float(reward),
                    done=done,
                    action_mask=mask_np,
                )
            )

            self._global_step += 1
            self._current_ep_return += float(reward)
            self._current_ep_length += 1

            if done:
                is_delivered = bool(info.get("reward_components", {}).get("delivery", 0.0) > 0.0)
                self._ep_deliveries.append(1.0 if is_delivered else 0.0)
                self._ep_returns.append(self._current_ep_return)
                self._ep_lengths.append(self._current_ep_length)
                self._current_ep_return = 0.0
                self._current_ep_length = 0
                next_obs, _ = self.env.reset()

            obs = next_obs

        return obs

    def _ppo_update(self) -> Dict[str, float]:
        """Run ``epochs_per_update`` gradient epochs over the current rollout batch.

        Returns:
            Dict of mean training statistics: ``policy_loss``, ``value_loss``,
            ``entropy``, ``approx_kl``, ``clip_fraction``.
        """
        self.policy.train()

        stats: Dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for epoch in range(self.cfg.epochs_per_update):
            epoch_kls: list[float] = []
            for batch in self.buffer.get_minibatches(
                minibatch_size=self.cfg.minibatch_size,
                normalize_advantages=True,
                return_action_masks=self.cfg.use_action_mask,
            ):
                if self.cfg.use_action_mask and len(batch) == 6:
                    obs_b, act_b, old_lp_b, ret_b, adv_b, mask_b = batch
                else:
                    obs_b, act_b, old_lp_b, ret_b, adv_b = batch[:5]
                    mask_b = None

                new_log_probs, values, entropy = self.policy.evaluate(
                    obs_b, act_b, action_mask=mask_b
                )

                # Clipped surrogate loss
                ratio = torch.exp(new_log_probs - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (clipped MSE)
                values_flat = values.squeeze(-1)
                value_loss = nn.functional.mse_loss(values_flat, ret_b)

                # Total loss
                loss = policy_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_range).float().mean().item()

                epoch_kls.append(approx_kl)
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(approx_kl)
                stats["clip_fraction"].append(clip_frac)

            # Target KL early stopping to prevent policy collapse over long runs
            mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if getattr(self.cfg, "target_kl", None) is not None and mean_epoch_kl > self.cfg.target_kl:
                logger.debug("Early stopping PPO epochs due to target KL reached", epoch=epoch, kl=round(mean_epoch_kl, 5))
                break

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _save_checkpoint(self, tag: str) -> Path:
        """Save a model checkpoint tagged with ``tag``.

        Args:
            tag: Descriptive string suffix for the checkpoint file name.

        Returns:
            Path to the saved checkpoint file.
        """
        ckpt_path = self._output_dir / f"ppo_{tag}.pt"
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self._global_step,
                "update_count": self._update_count,
                "ep_returns": self._ep_returns[-100:],
                "ep_lengths": self._ep_lengths[-100:],
                "ep_deliveries": self._ep_deliveries[-100:],
                "ppo_config": self.cfg.model_dump(),
            },
            ckpt_path,
        )
        logger.info("Checkpoint saved", path=str(ckpt_path), global_step=self._global_step)
        return ckpt_path

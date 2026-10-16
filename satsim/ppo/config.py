"""PPO hyperparameter configuration model.

Defines :class:`PPOConfig`, the single source of truth for all
Proximal Policy Optimisation training hyperparameters.  Every field
carries a validated default so users can start training with no extra
YAML — while still allowing full customisation through YAML overrides
or programmatic construction.

Public API
----------
- ``PPOConfig`` — Pydantic model with ``load_yaml`` / ``to_yaml`` helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class PPOConfig(BaseModel):
    """Hyperparameter configuration for the PPO trainer.

    Args:
        gat_checkpoint: Path to the frozen pre-trained GAT checkpoint (.pt).
        lstm_checkpoint: Path to the frozen pre-trained LSTM checkpoint (.pt).
        gat_scaler: Path to the GAT feature scaler (.pkl).
        lstm_scaler: Path to the LSTM feature scaler (.pkl).
        output_dir: Root directory for checkpoints and training logs.
        max_hops: Maximum hops per episode in SatelliteRoutingEnv.
        rollout_steps: Number of environment steps per rollout collection.
        gamma: Discount factor γ for future rewards.
        gae_lambda: λ for Generalised Advantage Estimation.
        epochs_per_update: Number of gradient epochs per rollout batch.
        minibatch_size: Number of transitions per gradient mini-batch.
        lr: Adam learning rate.
        clip_range: PPO clipped surrogate ε.
        value_loss_coef: Coefficient c₁ for the value function loss.
        entropy_coef: Coefficient c₂ for the entropy bonus.
        max_grad_norm: Global gradient-norm clipping threshold.
        hidden_dim: Width of hidden layers in the Actor-Critic MLP.
        num_hidden_layers: Number of shared hidden layers (≥1).
        total_timesteps: Total environment steps to train for.
        seed: Global random seed for reproducibility.
        log_interval: Log metrics every N rollout updates.
        save_interval: Save a checkpoint every N rollout updates.
        device: PyTorch device string ('cpu', 'cuda', 'cuda:0', …).
        obs_dim: Observation dimension fed to the Actor-Critic (default 256).
        num_actions: Number of discrete actions (satellites in constellation).
    """

    # ---------- Paths ----------
    gat_checkpoint: str = Field(
        default="artifacts/gat/spatial/gat_best.pt",
        description="Path to the frozen pre-trained GAT model checkpoint.",
    )
    lstm_checkpoint: str = Field(
        default="artifacts/lstm/lstm_best.pt",
        description="Path to the frozen pre-trained LSTM model checkpoint.",
    )
    gat_scaler: str = Field(
        default="artifacts/gat/spatial/feature_scaler.pkl",
        description="Path to the GAT feature scaler pickle file.",
    )
    lstm_scaler: str = Field(
        default="artifacts/lstm/feature_scaler.pkl",
        description="Path to the LSTM feature scaler pickle file.",
    )
    output_dir: str = Field(
        default="artifacts/ppo",
        description="Root output directory for PPO checkpoints and logs.",
    )

    # ---------- Environment ----------
    max_hops: int = Field(default=20, ge=1, description="Max hops per episode.")
    num_actions: int = Field(
        default=100,
        ge=1,
        description="Number of discrete actions (satellites). Must match constellation size.",
    )

    # ---------- Rollout ----------
    rollout_steps: int = Field(
        default=2048,
        ge=1,
        description="Environment steps collected per rollout before each update.",
    )
    gamma: float = Field(default=0.99, gt=0.0, lt=1.0, description="Discount factor γ.")
    gae_lambda: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0,
        description="λ for Generalised Advantage Estimation.",
    )

    # ---------- Optimisation ----------
    epochs_per_update: int = Field(
        default=5,
        ge=1,
        description="Number of gradient epochs per collected rollout.",
    )
    minibatch_size: int = Field(
        default=64,
        ge=1,
        description="Number of transitions per gradient mini-batch.",
    )
    lr: float = Field(default=1e-4, gt=0.0, description="Adam learning rate.")
    clip_range: float = Field(
        default=0.15,
        gt=0.0,
        description="PPO clipped surrogate objective ε.",
    )
    value_loss_coef: float = Field(
        default=0.5,
        gt=0.0,
        description="Coefficient c₁ for the value function MSE loss.",
    )
    entropy_coef: float = Field(
        default=0.001,
        ge=0.0,
        description="Coefficient c₂ for the entropy bonus.",
    )
    max_grad_norm: float = Field(
        default=0.5,
        gt=0.0,
        description="Global gradient-norm clipping threshold.",
    )
    target_kl: Optional[float] = Field(
        default=0.015,
        description="Target KL divergence for early stopping PPO epochs to prevent policy collapse.",
    )
    lr_annealing: bool = Field(
        default=True,
        description="Whether to linearly anneal learning rate toward 1e-5 across total_timesteps.",
    )

    # ---------- Shared Network ----------
    hidden_dim: int = Field(
        default=256,
        ge=1,
        description="Hidden-layer width for the Actor-Critic MLP backbone.",
    )
    num_hidden_layers: int = Field(
        default=2,
        ge=1,
        description="Number of shared hidden layers (≥ 1).",
    )

    # ---------- Training Budget ----------
    total_timesteps: int = Field(
        default=1_000_000,
        ge=1,
        description="Total environment interaction steps.",
    )
    seed: int = Field(default=42, description="Global random seed for reproducibility.")

    # ---------- Logging / Checkpointing ----------
    log_interval: int = Field(
        default=10,
        ge=1,
        description="Log training metrics every N rollout updates.",
    )
    save_interval: int = Field(
        default=50,
        ge=1,
        description="Save a model checkpoint every N rollout updates.",
    )
    device: str = Field(
        default="cpu",
        description="PyTorch device string ('cpu', 'cuda', 'cuda:0', …).",
    )

    # ---------- Observation & Action Masking ----------
    obs_dim: int = Field(
        default=256,
        ge=1,
        description=(
            "Observation dimension fed to the Actor-Critic. "
            "Default 256 = 128-dim GAT spatial embedding || 128-dim LSTM temporal embedding."
        ),
    )
    use_action_mask: bool = Field(
        default=True,
        description="Whether to mask out non-neighbor ISL links to prevent invalid hop drops.",
    )
    use_progress_reward: bool = Field(
        default=True,
        description="Whether to add potential-based distance progress reward towards destination.",
    )
    include_dst_embedding: bool = Field(
        default=True,
        description="Whether to condition observation on destination satellite spatial embedding.",
    )
    prevent_loops: bool = Field(
        default=True,
        description="Whether to mask out already-visited satellites in the episode to eliminate routing loops.",
    )

    @classmethod
    def load_yaml(cls, path: str | Path) -> PPOConfig:
        """Load a :class:`PPOConfig` from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated :class:`PPOConfig` instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"PPOConfig YAML not found at '{path}'. "
                "Create the file or pass a valid path."
            )
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data or {})

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this config to a YAML file.

        Args:
            path: Destination path for the YAML file.

        Returns:
            None
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(self.model_dump(), fh, sort_keys=False)

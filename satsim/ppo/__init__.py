"""satsim.ppo — Proximal Policy Optimisation module for satellite routing.

This package provides all components needed to train and deploy a PPO agent
on the ``SatelliteRouting-v0`` Gymnasium environment with a 256-dimensional
fused observation derived from pre-trained frozen GAT and LSTM models.

Submodules
----------
- :mod:`satsim.ppo.config`         — :class:`PPOConfig` hyperparameter model.
- :mod:`satsim.ppo.actor_critic`   — :class:`PPOActorCritic` network.
- :mod:`satsim.ppo.rollout_buffer` — :class:`RolloutBuffer` with GAE.
- :mod:`satsim.ppo.obs_wrapper`    — :class:`FusedEmbeddingWrapper` (GAT+LSTM → 256-dim obs).
- :mod:`satsim.ppo.trainer`        — :class:`PPOTrainer` training loop.
"""
from __future__ import annotations

from satsim.ppo.config import PPOConfig
from satsim.ppo.actor_critic import PPOActorCritic
from satsim.ppo.rollout_buffer import RolloutBuffer, Transition
from satsim.ppo.obs_wrapper import FusedEmbeddingWrapper
from satsim.ppo.trainer import PPOTrainer

__all__ = [
    "PPOConfig",
    "PPOActorCritic",
    "RolloutBuffer",
    "Transition",
    "FusedEmbeddingWrapper",
    "PPOTrainer",
]

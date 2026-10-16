"""PPO Actor-Critic neural network for satellite routing.

Provides :class:`PPOActorCritic` — a shared-backbone MLP that processes
the 256-dimensional fused GAT+LSTM observation and outputs:

* A probability distribution over discrete next-hop actions (policy head).
* A scalar state-value estimate (value head).

Public API
----------
- ``PPOActorCritic`` — ``nn.Module`` with ``act`` and ``evaluate`` helpers.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    num_hidden: int,
) -> nn.Sequential:
    """Build a multi-layer ReLU MLP trunk.

    Args:
        in_dim: Input feature dimensionality.
        hidden_dim: Width of each hidden layer.
        num_hidden: Number of hidden layers (≥ 1).

    Returns:
        ``nn.Sequential`` containing Linear → ReLU × num_hidden blocks.
    """
    layers: List[nn.Module] = []
    current_dim = in_dim
    for _ in range(num_hidden):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class PPOActorCritic(nn.Module):
    """Shared-backbone Actor-Critic network for PPO on ``SatelliteRouting-v0``.

    Architecture
    ------------
    Shared trunk: Linear(obs_dim → hidden_dim) → ReLU repeated ``num_hidden_layers`` times.
    Policy head:  Linear(hidden_dim → num_actions) [outputs logits].
    Value head:   Linear(hidden_dim → 1) [outputs scalar state-value].

    The 256-dim observation is the concatenation of:
    * GAT spatial embedding for ``curr_node`` [128-dim]
    * LSTM temporal embedding for ``curr_node`` [128-dim]

    Args:
        obs_dim: Observation dimensionality (default 256).
        num_actions: Size of the discrete action space (default 100 satellites).
        hidden_dim: Hidden layer width (default 256).
        num_hidden_layers: Number of shared hidden layers (default 2).
    """

    def __init__(
        self,
        obs_dim: int = 256,
        num_actions: int = 100,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Shared backbone
        self.shared = _build_mlp(obs_dim, hidden_dim, num_hidden_layers)

        # Policy (actor) head: outputs unnormalised logits
        self.policy_head = nn.Linear(hidden_dim, num_actions)

        # Value (critic) head: outputs scalar estimate
        self.value_head = nn.Linear(hidden_dim, 1)

        # Orthogonal weight initialisation (common PPO best-practice)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialisation to all linear layers.

        Returns:
            None
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        # Policy head gets a smaller gain to start with more uniform logits
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute policy logits and state-value estimates.

        Args:
            obs: Fused observation tensor of shape ``[batch, obs_dim]``.

        Returns:
            Tuple of:
                - ``logits``: Policy logits ``[batch, num_actions]``.
                - ``value``: State-value estimates ``[batch, 1]``.
        """
        features = self.shared(obs)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or deterministically select) an action given an observation and optional action mask.

        Args:
            obs: Observation tensor ``[batch, obs_dim]`` or ``[obs_dim]``.
            action_mask: Optional boolean mask ``[batch, num_actions]`` or ``[num_actions]``.
                         True indicates valid actions (e.g. connected ISL neighbors).
            deterministic: If ``True``, return the argmax action (greedy).

        Returns:
            Tuple of:
                - ``action``: Selected action ``[batch]`` (integer indices).
                - ``log_prob``: Log-probability of the selected action ``[batch]``.
                - ``value``: State-value estimate ``[batch, 1]``.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits, value = self.forward(obs)

        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e8)

        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions under the current policy (used in PPO update).

        Args:
            obs: Batch of observations ``[batch, obs_dim]``.
            actions: Batch of previously taken actions ``[batch]``.
            action_mask: Optional boolean mask ``[batch, num_actions]``.

        Returns:
            Tuple of:
                - ``log_probs``: Log-probabilities of the given actions ``[batch]``.
                - ``values``: Current value estimates ``[batch, 1]``.
                - ``entropy``: Mean entropy of the policy distribution ``[scalar]``.
        """
        logits, values = self.forward(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, values, entropy

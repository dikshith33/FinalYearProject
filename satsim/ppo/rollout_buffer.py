"""Rollout buffer for PPO with Generalised Advantage Estimation (GAE).

Stores a fixed-length trajectory of environment transitions and computes
discounted returns and GAE advantages before each policy gradient update.

Public API
----------
- ``Transition`` — dataclass holding one env step's data.
- ``RolloutBuffer`` — fixed-capacity buffer with ``add``, ``finish_episode``,
  ``compute_returns_and_advantages``, and ``get_minibatches`` helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    """Single environment transition stored in the rollout buffer.

    Args:
        obs: Observation vector at time t (shape ``[obs_dim]``).
        action: Integer action taken at time t.
        log_prob: Log-probability of the taken action under the old policy.
        value: State-value estimate at time t.
        reward: Scalar reward received after the action.
        done: Whether the episode terminated or was truncated after this step.
        action_mask: Optional boolean mask of valid actions at time t.
    """

    obs: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    action_mask: np.ndarray | None = None


class RolloutBuffer:
    """Fixed-capacity rollout buffer with GAE advantage computation.

    Collects up to ``capacity`` transitions, then computes discounted
    bootstrapped returns and GAE advantages for the entire batch.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Dimensionality of the observation vector.
        gamma: Discount factor γ (default 0.99).
        gae_lambda: GAE λ parameter (default 0.95).
        device: PyTorch device for tensor outputs.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        self._observations: List[np.ndarray] = []
        self._actions: List[int] = []
        self._log_probs: List[float] = []
        self._values: List[float] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._action_masks: List[np.ndarray | None] = []

        # Filled by compute_returns_and_advantages
        self._returns: np.ndarray | None = None
        self._advantages: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def add(self, transition: Transition) -> None:
        """Append one transition to the buffer.

        Args:
            transition: The :class:`Transition` to store.

        Returns:
            None
        """
        if len(self._observations) >= self.capacity:
            raise ValueError(
                f"RolloutBuffer is full ({self.capacity} transitions). "
                "Call compute_returns_and_advantages() then clear() before adding more."
            )
        self._observations.append(transition.obs.copy())
        self._actions.append(transition.action)
        self._log_probs.append(transition.log_prob)
        self._values.append(transition.value)
        self._rewards.append(transition.reward)
        self._dones.append(transition.done)
        self._action_masks.append(
            transition.action_mask.copy() if transition.action_mask is not None else None
        )

    def __len__(self) -> int:
        """Return the number of transitions currently stored.

        Returns:
            Integer count of stored transitions.
        """
        return len(self._observations)

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached its capacity.

        Returns:
            True if capacity is reached.
        """
        return len(self) >= self.capacity

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """Compute bootstrapped returns and GAE advantages in-place.

        Uses the Generalised Advantage Estimation (GAE-λ) formula:

            δₜ  = rₜ + γ · V(sₜ₊₁) · (1 − doneₜ) − V(sₜ)
            Aₜ  = δₜ + (γ · λ)^1 · δₜ₊₁ + … (backwards recursion)
            Gₜ  = Aₜ + V(sₜ)

        Args:
            last_value: Bootstrap value V(s_{T+1}) from the critic.
                        Pass 0.0 if the last step was terminal.

        Returns:
            None
        """
        n = len(self._observations)
        if n == 0:
            raise ValueError(
                "Cannot compute returns on an empty buffer. "
                "Add at least one transition first."
            )

        rewards = np.array(self._rewards, dtype=np.float32)
        values = np.array(self._values, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_non_terminal = 1.0 - (dones[t])
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        self._advantages = advantages
        self._returns = advantages + values

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def get_minibatches(
        self,
        minibatch_size: int,
        normalize_advantages: bool = True,
        return_action_masks: bool = False,
    ) -> Generator[
        Tuple[Any, ...],
        None,
        None,
    ]:
        """Yield shuffled mini-batches for the PPO gradient update.

        Args:
            minibatch_size: Size of each mini-batch.
            normalize_advantages: If True, normalise advantages to zero mean /
                unit variance within the full batch before splitting.
            return_action_masks: If True, also yield boolean action masks tensor.

        Yields:
            Tuples of ``(obs, actions, old_log_probs, returns, advantages)``
            or ``(obs, actions, old_log_probs, returns, advantages, action_masks)``.
        """
        if self._returns is None or self._advantages is None:
            raise ValueError(
                "Call compute_returns_and_advantages() before get_minibatches()."
            )

        n = len(self._observations)
        obs_arr = np.array(self._observations, dtype=np.float32)
        actions_arr = np.array(self._actions, dtype=np.int64)
        log_probs_arr = np.array(self._log_probs, dtype=np.float32)
        advantages = self._advantages.copy()

        has_masks = any(m is not None for m in self._action_masks)
        masks_arr: np.ndarray | None = None
        if has_masks:
            # Replace any None masks with True (all valid)
            first_mask = next(m for m in self._action_masks if m is not None)
            num_actions = len(first_mask)
            filled_masks = [
                m if m is not None else np.ones(num_actions, dtype=bool)
                for m in self._action_masks
            ]
            masks_arr = np.array(filled_masks, dtype=bool)

        if normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        returns = self._returns

        indices = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            obs_b = torch.tensor(obs_arr[batch_idx], dtype=torch.float32, device=self.device)
            act_b = torch.tensor(actions_arr[batch_idx], dtype=torch.long, device=self.device)
            lp_b = torch.tensor(log_probs_arr[batch_idx], dtype=torch.float32, device=self.device)
            ret_b = torch.tensor(returns[batch_idx], dtype=torch.float32, device=self.device)
            adv_b = torch.tensor(advantages[batch_idx], dtype=torch.float32, device=self.device)

            if return_action_masks:
                if masks_arr is not None:
                    mask_b = torch.tensor(masks_arr[batch_idx], dtype=torch.bool, device=self.device)
                else:
                    mask_b = None
                yield (obs_b, act_b, lp_b, ret_b, adv_b, mask_b)
            else:
                yield (obs_b, act_b, lp_b, ret_b, adv_b)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset the buffer, discarding all stored transitions and computed values.

        Returns:
            None
        """
        self._observations.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()
        self._action_masks.clear()
        self._returns = None
        self._advantages = None

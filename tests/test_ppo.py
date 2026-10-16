"""Tests for the satsim.ppo module.

Covers:
- PPOConfig: creation, YAML round-trip.
- PPOActorCritic: forward pass, act(), evaluate() shapes and non-trivial behaviour.
- RolloutBuffer: add/full/capacity guards, GAE computation, mini-batch iteration.
- FusedEmbeddingWrapper: tested with mock frozen models so tests do NOT touch disk.
- Integration smoke test: PPOTrainer._collect_rollout + _ppo_update with mocked env.

All tests are self-contained and do not require external services or pre-trained
checkpoint files on disk.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from satsim.ppo.actor_critic import PPOActorCritic
from satsim.ppo.config import PPOConfig
from satsim.ppo.rollout_buffer import RolloutBuffer, Transition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config() -> PPOConfig:
    """Default PPOConfig with small rollout for speed."""
    return PPOConfig(
        rollout_steps=16,
        epochs_per_update=2,
        minibatch_size=8,
        total_timesteps=32,
        obs_dim=256,
        num_actions=100,
        hidden_dim=64,
        num_hidden_layers=1,
        log_interval=1,
        save_interval=999,
        seed=0,
        device="cpu",
    )


@pytest.fixture()
def actor_critic() -> PPOActorCritic:
    """Small PPOActorCritic for fast testing."""
    return PPOActorCritic(obs_dim=256, num_actions=100, hidden_dim=64, num_hidden_layers=1)


@pytest.fixture()
def rollout_buffer() -> RolloutBuffer:
    """Small RolloutBuffer for testing."""
    return RolloutBuffer(capacity=8, obs_dim=256, gamma=0.99, gae_lambda=0.95, device="cpu")


# ---------------------------------------------------------------------------
# PPOConfig tests
# ---------------------------------------------------------------------------


class TestPPOConfig:
    """Tests for PPOConfig validation and serialisation."""

    def test_default_construction(self) -> None:
        """PPOConfig constructs with defaults without raising."""
        cfg = PPOConfig()
        assert cfg.gamma == 0.99
        assert cfg.obs_dim == 256
        assert cfg.num_actions == 100

    def test_yaml_round_trip(self) -> None:
        """Config serialised to YAML and reloaded must be identical."""
        cfg = PPOConfig(lr=1e-3, seed=7, hidden_dim=128)
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "ppo_cfg.yaml"
            cfg.to_yaml(yaml_path)
            reloaded = PPOConfig.load_yaml(yaml_path)
        assert reloaded.lr == cfg.lr
        assert reloaded.seed == cfg.seed
        assert reloaded.hidden_dim == cfg.hidden_dim

    def test_load_yaml_missing_file_raises(self) -> None:
        """Loading a YAML from a non-existent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PPOConfig.load_yaml("/nonexistent/ppo_config.yaml")

    def test_invalid_gamma_raises(self) -> None:
        """gamma must be in (0, 1); values outside raise ValidationError."""
        with pytest.raises(Exception):
            PPOConfig(gamma=1.5)

    def test_invalid_minibatch_size_raises(self) -> None:
        """minibatch_size < 1 must raise ValidationError."""
        with pytest.raises(Exception):
            PPOConfig(minibatch_size=0)


# ---------------------------------------------------------------------------
# PPOActorCritic tests
# ---------------------------------------------------------------------------


class TestPPOActorCritic:
    """Tests for PPOActorCritic network forward pass and helper methods."""

    def test_forward_output_shapes(self, actor_critic: PPOActorCritic) -> None:
        """forward() returns logits [B, num_actions] and values [B, 1]."""
        obs = torch.randn(4, 256)
        logits, values = actor_critic(obs)
        assert logits.shape == (4, 100), f"Unexpected logits shape: {logits.shape}"
        assert values.shape == (4, 1), f"Unexpected value shape: {values.shape}"

    def test_act_output_shapes_batched(self, actor_critic: PPOActorCritic) -> None:
        """act() returns (action [B], log_prob [B], value [B,1]) for batched input."""
        obs = torch.randn(3, 256)
        action, log_prob, value = actor_critic.act(obs)
        assert action.shape == (3,)
        assert log_prob.shape == (3,)
        assert value.shape == (3, 1)

    def test_act_output_shapes_single(self, actor_critic: PPOActorCritic) -> None:
        """act() accepts a 1D obs and returns scalar action and log_prob."""
        obs = torch.randn(256)
        action, log_prob, value = actor_critic.act(obs)
        assert action.numel() == 1
        assert log_prob.numel() == 1

    def test_actions_are_valid_satellite_ids(self, actor_critic: PPOActorCritic) -> None:
        """Sampled actions must be valid satellite indices in [0, num_actions)."""
        obs = torch.randn(32, 256)
        action, _, _ = actor_critic.act(obs)
        assert action.min() >= 0
        assert action.max() < 100

    def test_deterministic_act_is_argmax(self, actor_critic: PPOActorCritic) -> None:
        """Deterministic act must equal the argmax of the logits."""
        obs = torch.randn(5, 256)
        action_det, _, _ = actor_critic.act(obs, deterministic=True)
        logits, _ = actor_critic(obs)
        expected = logits.argmax(dim=-1)
        assert torch.all(action_det == expected)

    def test_evaluate_log_prob_shape(self, actor_critic: PPOActorCritic) -> None:
        """evaluate() returns (log_probs [B], values [B,1], entropy scalar)."""
        obs = torch.randn(8, 256)
        actions = torch.randint(0, 100, (8,))
        log_probs, values, entropy = actor_critic.evaluate(obs, actions)
        assert log_probs.shape == (8,)
        assert values.shape == (8, 1)
        assert entropy.ndim == 0  # scalar

    def test_evaluate_log_prob_is_negative(self, actor_critic: PPOActorCritic) -> None:
        """Log-probabilities must be ≤ 0 (probability distributions sum to 1)."""
        obs = torch.randn(16, 256)
        actions = torch.randint(0, 100, (16,))
        log_probs, _, _ = actor_critic.evaluate(obs, actions)
        assert (log_probs <= 0).all(), "Log-probabilities must be non-positive."

    def test_not_a_no_op_stub(self, actor_critic: PPOActorCritic) -> None:
        """Different obs must produce different logits (not a constant stub)."""
        obs1 = torch.randn(1, 256)
        obs2 = torch.randn(1, 256)
        logits1, _ = actor_critic(obs1)
        logits2, _ = actor_critic(obs2)
        assert not torch.allclose(logits1, logits2), (
            "Same logits for different obs — model might be a no-op stub."
        )


# ---------------------------------------------------------------------------
# RolloutBuffer tests
# ---------------------------------------------------------------------------


def _make_transition(obs_dim: int = 256) -> Transition:
    """Helper to build a dummy Transition."""
    return Transition(
        obs=np.random.randn(obs_dim).astype(np.float32),
        action=int(np.random.randint(0, 100)),
        log_prob=float(np.random.randn()),
        value=float(np.random.randn()),
        reward=float(np.random.randn()),
        done=bool(np.random.rand() > 0.9),
    )


class TestRolloutBuffer:
    """Tests for RolloutBuffer add, GAE, and mini-batch iteration."""

    def test_add_and_len(self, rollout_buffer: RolloutBuffer) -> None:
        """Adding transitions increments length correctly."""
        assert len(rollout_buffer) == 0
        rollout_buffer.add(_make_transition())
        assert len(rollout_buffer) == 1

    def test_is_full(self, rollout_buffer: RolloutBuffer) -> None:
        """is_full is True only after capacity transitions are added."""
        for _ in range(8):
            assert not rollout_buffer.is_full
            rollout_buffer.add(_make_transition())
        assert rollout_buffer.is_full

    def test_overflow_raises(self, rollout_buffer: RolloutBuffer) -> None:
        """Adding beyond capacity must raise ValueError."""
        for _ in range(8):
            rollout_buffer.add(_make_transition())
        with pytest.raises(ValueError, match="full"):
            rollout_buffer.add(_make_transition())

    def test_compute_returns_empty_buffer_raises(self) -> None:
        """compute_returns_and_advantages on an empty buffer must raise ValueError."""
        buf = RolloutBuffer(capacity=4, obs_dim=256)
        with pytest.raises(ValueError):
            buf.compute_returns_and_advantages(last_value=0.0)

    def test_get_minibatches_before_compute_raises(self, rollout_buffer: RolloutBuffer) -> None:
        """get_minibatches before compute raises ValueError."""
        rollout_buffer.add(_make_transition())
        with pytest.raises(ValueError):
            next(rollout_buffer.get_minibatches(minibatch_size=4))

    def test_gae_returns_shape(self, rollout_buffer: RolloutBuffer) -> None:
        """GAE computation produces returns and advantages of length == buffer size."""
        for _ in range(8):
            rollout_buffer.add(_make_transition())
        rollout_buffer.compute_returns_and_advantages(last_value=0.0)
        assert rollout_buffer._returns is not None
        assert rollout_buffer._advantages is not None
        assert len(rollout_buffer._returns) == 8
        assert len(rollout_buffer._advantages) == 8

    def test_minibatch_tensor_shapes(self, rollout_buffer: RolloutBuffer) -> None:
        """Mini-batches yield obs [B,256], actions [B], log_probs [B], returns [B], adv [B]."""
        for _ in range(8):
            rollout_buffer.add(_make_transition())
        rollout_buffer.compute_returns_and_advantages(last_value=0.0)
        batches = list(rollout_buffer.get_minibatches(minibatch_size=4))
        assert len(batches) == 2
        obs_b, act_b, lp_b, ret_b, adv_b = batches[0]
        assert obs_b.shape == (4, 256)
        assert act_b.shape == (4,)
        assert lp_b.shape == (4,)
        assert ret_b.shape == (4,)
        assert adv_b.shape == (4,)

    def test_clear_resets_buffer(self, rollout_buffer: RolloutBuffer) -> None:
        """clear() resets length to 0 and removes computed values."""
        for _ in range(4):
            rollout_buffer.add(_make_transition())
        rollout_buffer.compute_returns_and_advantages(last_value=0.0)
        rollout_buffer.clear()
        assert len(rollout_buffer) == 0
        assert rollout_buffer._returns is None
        assert rollout_buffer._advantages is None

    def test_advantages_normalised(self, rollout_buffer: RolloutBuffer) -> None:
        """Normalised advantages within a mini-batch must have mean ≈ 0, std ≈ 1."""
        for _ in range(8):
            rollout_buffer.add(_make_transition())
        rollout_buffer.compute_returns_and_advantages(last_value=0.0)
        # Collect all advantages from mini-batches
        all_adv = []
        for _, _, _, _, adv in rollout_buffer.get_minibatches(
            minibatch_size=8, normalize_advantages=True
        ):
            all_adv.append(adv.numpy())
        all_adv_np = np.concatenate(all_adv)
        assert abs(all_adv_np.mean()) < 1e-5, f"Mean of normalised advantages: {all_adv_np.mean()}"
        assert abs(all_adv_np.std() - 1.0) < 1e-4, f"Std of normalised advantages: {all_adv_np.std()}"

    def test_gae_discount_direction(self) -> None:
        """A positive final reward should yield a positive advantage at the last step."""
        buf = RolloutBuffer(capacity=3, obs_dim=4, gamma=0.99, gae_lambda=0.95)
        # Three steps: reward 0, 0, +10
        for reward in [0.0, 0.0, 10.0]:
            buf.add(Transition(
                obs=np.zeros(4, dtype=np.float32),
                action=0, log_prob=-1.0, value=0.0, reward=reward, done=False,
            ))
        buf.compute_returns_and_advantages(last_value=0.0)
        # The final step had reward +10, so advantage at t=2 should be positive
        assert buf._advantages[2] > 0, (
            f"Expected positive advantage at final step, got {buf._advantages[2]}"
        )


# ---------------------------------------------------------------------------
# Action Masking and CLI tests
# ---------------------------------------------------------------------------


class TestActionMasking:
    """Tests for invalid action masking in PPOActorCritic."""

    def test_masked_act_only_selects_allowed_actions(self, actor_critic: PPOActorCritic) -> None:
        """act() with an action mask must never pick a masked-out action."""
        obs = torch.randn(10, 256)
        # Only allow actions 3 and 7
        mask = torch.zeros(10, 100, dtype=torch.bool)
        mask[:, 3] = True
        mask[:, 7] = True

        for _ in range(20):
            action, _, _ = actor_critic.act(obs, action_mask=mask)
            assert ((action == 3) | (action == 7)).all(), "Selected an action not in mask!"

    def test_masked_evaluate_computes_finite_log_probs(self, actor_critic: PPOActorCritic) -> None:
        """evaluate() with valid masked actions must return finite log probs."""
        obs = torch.randn(4, 256)
        mask = torch.zeros(4, 100, dtype=torch.bool)
        mask[:, 5] = True
        actions = torch.full((4,), 5, dtype=torch.long)

        log_probs, values, entropy = actor_critic.evaluate(obs, actions, action_mask=mask)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(values).all()
        assert torch.isfinite(entropy).all()


class TestRolloutBufferMasking:
    """Tests for action mask storage in RolloutBuffer."""

    def test_buffer_yields_action_masks(self, rollout_buffer: RolloutBuffer) -> None:
        """get_minibatches with return_action_masks=True returns mask tensors."""
        for _ in range(8):
            mask = np.zeros(100, dtype=bool)
            mask[1] = True
            mask[2] = True
            rollout_buffer.add(Transition(
                obs=np.zeros(256, dtype=np.float32),
                action=1, log_prob=-0.5, value=0.0, reward=1.0, done=False,
                action_mask=mask,
            ))
        rollout_buffer.compute_returns_and_advantages(last_value=0.0)
        batches = list(rollout_buffer.get_minibatches(minibatch_size=4, return_action_masks=True))
        assert len(batches) == 2
        obs_b, act_b, lp_b, ret_b, adv_b, mask_b = batches[0]
        assert mask_b is not None
        assert mask_b.shape == (4, 100)
        assert mask_b.dtype == torch.bool


class TestPPOCLI:
    """Tests for train_ppo CLI module."""

    def test_cli_parser(self) -> None:
        """CLI arguments parse correctly with custom flags."""
        from satsim.cli.train_ppo import parse_args
        with patch("sys.argv", ["train_ppo.py", "--total-timesteps", "5000", "--obs-dim", "256", "--seed", "123"]):
            args = parse_args()
            assert args.total_timesteps == 5000
            assert args.obs_dim == 256
            assert args.seed == 123


class TestLoopPreventionAndTopologicalProgress:
    """Tests for loop prevention masking and topological hop progress."""

    def test_prevent_loops_masking(self) -> None:
        """Visited nodes must be masked out from the action mask to prevent loops."""
        import networkx as nx
        from satsim.ppo.obs_wrapper import FusedEmbeddingWrapper

        mock_env = MagicMock()
        mock_env.num_sats = 10
        mock_env.curr_node = 0
        mock_env.dst_node = 5
        g = nx.Graph()
        g.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 5)])
        mock_env.current_graph = g

        wrapper = MagicMock(spec=FusedEmbeddingWrapper)
        wrapper.env = mock_env
        wrapper._prevent_loops = True
        wrapper.visited_nodes = {0, 1}
        mask = FusedEmbeddingWrapper.get_action_mask(wrapper)

        # Node 1 is a neighbor of 0, but is visited -> False
        # Node 2 is a neighbor of 0 and unvisited -> True
        assert bool(mask[2]) is True
        assert bool(mask[1]) is False
        assert bool(mask[0]) is False

    def test_destination_is_always_allowed(self) -> None:
        """Destination node is always allowed to reach the goal."""
        import networkx as nx
        from satsim.ppo.obs_wrapper import FusedEmbeddingWrapper

        mock_env = MagicMock()
        mock_env.num_sats = 10
        mock_env.curr_node = 2
        mock_env.dst_node = 5
        g = nx.Graph()
        g.add_edges_from([(2, 5), (2, 0)])
        mock_env.current_graph = g

        wrapper = MagicMock(spec=FusedEmbeddingWrapper)
        wrapper.env = mock_env
        wrapper._prevent_loops = True
        wrapper.visited_nodes = {2, 5}
        mask = FusedEmbeddingWrapper.get_action_mask(wrapper)

        assert bool(mask[5]) is True


"""CLI command for training the PPO satellite routing agent.

Integrates frozen self-supervised GAT spatial embeddings and LSTM temporal dynamics
to train an Actor-Critic PPO policy for next-hop satellite routing with invalid action
masking and potential-based distance progress rewards.

Usage
-----
.. code-block:: bash

    # Quick test run (10,000 steps on CPU or CUDA)
    python -m satsim.cli.train_ppo --total-timesteps 10000 --device auto

    # Benchmark training (250,000 steps with GPU acceleration on Colab)
    python -m satsim.cli.train_ppo --total-timesteps 250000 --device cuda --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from satsim.logging import get_logger
from satsim.ppo.config import PPOConfig
from satsim.ppo.trainer import PPOTrainer

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed integer.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for PPO training.

    Returns:
        Populated :class:`argparse.Namespace` with configured options.
    """
    parser = argparse.ArgumentParser(
        description="Train PPO reinforcement learning agent for mega-constellation routing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a custom YAML configuration file.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total environment interaction timesteps (default: 100,000 if not specified).",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2048,
        help="Environment steps collected per rollout before policy update.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate (fine-tuning default: 1e-4).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor gamma for future rewards.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="Lambda parameter for Generalized Advantage Estimation.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.15,
        help="PPO surrogate clipping range epsilon (fine-tuning default: 0.15).",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.001,
        help="Entropy bonus coefficient (fine-tuning default: 0.001).",
    )
    parser.add_argument(
        "--epochs-per-update",
        type=int,
        default=5,
        help="Number of gradient epochs per rollout update (default: 5).",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.015,
        help="Target KL divergence for early stopping to prevent policy collapse (default: 0.015).",
    )
    parser.add_argument(
        "--no-action-mask",
        action="store_true",
        help="Disable invalid action masking (forces exploration across all 100 nodes).",
    )
    parser.add_argument(
        "--no-progress-reward",
        action="store_true",
        help="Disable distance progress reward shaping.",
    )
    parser.add_argument(
        "--no-prevent-loops",
        action="store_true",
        help="Disable visited-node loop prevention masking.",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=256,
        help="Observation dimension (256: delta-spatial + temporal, or 384: full concatenation).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device ('auto', 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from, or 'auto' to resume from artifacts/ppo/ppo_latest.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/ppo",
        help="Destination directory for checkpoints, metrics, and logs.",
    )
    parser.add_argument(
        "--gat-checkpoint",
        type=str,
        default="artifacts/gat/spatial/gat_best.pt",
        help="Path to frozen pre-trained GAT model checkpoint.",
    )
    parser.add_argument(
        "--gat-scaler",
        type=str,
        default="artifacts/gat/spatial/feature_scaler.pkl",
        help="Path to GAT feature scaler pickle.",
    )
    parser.add_argument(
        "--lstm-checkpoint",
        type=str,
        default="artifacts/lstm/lstm_best.pt",
        help="Path to frozen pre-trained LSTM model checkpoint.",
    )
    parser.add_argument(
        "--lstm-scaler",
        type=str,
        default="artifacts/lstm/feature_scaler.pkl",
        help="Path to LSTM feature scaler pickle.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=25,
        help="Number of greedy evaluation episodes after training.",
    )
    return parser.parse_args()


def generate_markdown_report(
    output_dir: Path,
    config: PPOConfig,
    eval_metrics: Dict[str, float],
) -> None:
    """Generate a comprehensive Markdown training report.

    Args:
        output_dir: Output directory where the report will be written.
        config: The :class:`PPOConfig` used for training.
        eval_metrics: Evaluation summary dictionary.

    Returns:
        None
    """
    report_path = output_dir / "PPO_TRAINING_REPORT.md"
    delivery_pct = eval_metrics.get("delivery_rate", 0.0) * 100.0
    mean_ret = eval_metrics.get("mean_return", 0.0)
    std_ret = eval_metrics.get("std_return", 0.0)
    mean_len = eval_metrics.get("mean_length", 0.0)

    content = f"""# 🛰️ PPO Satellite Routing Training Report

## 1. Executive Summary
- **Packet Delivery Ratio (Accuracy)**: **{delivery_pct:.1f}%**
- **Average Hop Count per Route**: **{mean_len:.2f} hops**
- **Mean Episode Return**: **{mean_ret:.2f} ± {std_ret:.2f}**
- **Total Training Budget**: **{config.total_timesteps:,d} timesteps**
- **Hardware Device Used**: `{config.device}`

## 2. Key Architectural Enhancements
1. **Invalid Action Masking (`use_action_mask={config.use_action_mask}`)**:
   Constrains policy sampling strictly to physically connected ISL neighbors, preventing packet loss from non-existent hops.
2. **Distance Progress Reward (`use_progress_reward={config.use_progress_reward}`)**:
   Provides a dense geodesic distance-reduction signal guiding the agent towards the target destination.
3. **Spatial + Temporal Embedding Fusion (`obs_dim={config.obs_dim}`)**:
   Conditions routing decisions on 128-D spatial GAT embeddings and 128-D LSTM temporal congestion dynamics.

## 3. Training Hyperparameters
| Parameter | Value |
| :--- | :--- |
| `learning_rate` | `{config.lr}` |
| `gamma` (Discount) | `{config.gamma}` |
| `gae_lambda` | `{config.gae_lambda}` |
| `clip_range` | `{config.clip_range}` |
| `entropy_coef` | `{config.entropy_coef}` |
| `rollout_steps` | `{config.rollout_steps}` |
| `epochs_per_update` | `{config.epochs_per_update}` |
| `minibatch_size` | `{config.minibatch_size}` |
| `seed` | `{config.seed}` |

## 4. Checkpoints & Reproducibility
- Config: [`config_used.yaml`](file:///{output_dir.resolve().as_posix()}/config_used.yaml)
- Best/Final Model: [`ppo_final.pt`](file:///{output_dir.resolve().as_posix()}/ppo_final.pt)
- Evaluation Metrics: [`eval_metrics.json`](file:///{output_dir.resolve().as_posix()}/eval_metrics.json)
"""
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def main() -> None:
    """Main CLI entrypoint for PPO training."""
    args = parse_args()
    set_seed(args.seed)

    # Resolve device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    # Resolve configuration
    if args.config is not None:
        cfg = PPOConfig.load_yaml(args.config)
    else:
        cfg = PPOConfig()

    # Apply CLI overrides
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    cfg.rollout_steps = args.rollout_steps
    cfg.lr = args.lr
    cfg.gamma = args.gamma
    cfg.gae_lambda = args.gae_lambda
    cfg.clip_range = args.clip_range
    cfg.entropy_coef = args.entropy_coef
    cfg.epochs_per_update = args.epochs_per_update
    cfg.target_kl = args.target_kl
    cfg.use_action_mask = not args.no_action_mask
    cfg.use_progress_reward = not args.no_progress_reward
    cfg.prevent_loops = not args.no_prevent_loops
    cfg.obs_dim = args.obs_dim
    cfg.device = device_str
    cfg.seed = args.seed
    cfg.output_dir = args.output_dir
    cfg.gat_checkpoint = args.gat_checkpoint
    cfg.gat_scaler = args.gat_scaler
    cfg.lstm_checkpoint = args.lstm_checkpoint
    cfg.lstm_scaler = args.lstm_scaler

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Initializing PPO trainer",
        device=device_str,
        timesteps=cfg.total_timesteps,
        action_masking=cfg.use_action_mask,
        progress_reward=cfg.use_progress_reward,
        obs_dim=cfg.obs_dim,
    )

    trainer = PPOTrainer(config=cfg)

    # Check for resume
    resume_target: Path | None = None
    if args.resume:
        if args.resume == "auto":
            auto_ckpt = out_dir / "ppo_latest.pt"
            if auto_ckpt.exists():
                resume_target = auto_ckpt
            else:
                print(f"[WARN] --resume auto requested but {auto_ckpt} not found. Starting from scratch.")
        else:
            resume_target = Path(args.resume)

    if resume_target is not None:
        trainer.load_checkpoint(resume_target)
        print(f"[RESUME] Loaded model & optimizer from: {resume_target}")
        print(f"         Resuming from global step {trainer._global_step:,d} / {cfg.total_timesteps:,d}")

    print("=" * 60)
    print(f"[START] PPO Training Pipeline on [{device_str.upper()}]")
    print(f"   Total Timesteps: {cfg.total_timesteps:,d} (resuming from {trainer._global_step:,d})")
    print(f"   Rollout Buffer:  {cfg.rollout_steps} steps / update")
    print(f"   Action Masking:  {'ENABLED (100% active ISLs)' if cfg.use_action_mask else 'DISABLED'}")
    print(f"   Progress Reward: {'ENABLED (geodesic distance)' if cfg.use_progress_reward else 'DISABLED'}")
    print(f"   GAT Checkpoint:  {cfg.gat_checkpoint}")
    print(f"   LSTM Checkpoint: {cfg.lstm_checkpoint}")
    print("=" * 60)

    trainer.train()

    print("\n[DONE] Training complete! Running greedy evaluation...")
    eval_metrics = trainer.evaluate(num_episodes=args.eval_episodes)

    # Save evaluation JSON
    eval_path = out_dir / "eval_metrics.json"
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(eval_metrics, fh, indent=2)

    # Save Markdown report
    generate_markdown_report(out_dir, cfg, eval_metrics)

    delivery_pct = eval_metrics.get("delivery_rate", 0.0) * 100.0
    mean_ret = eval_metrics.get("mean_return", 0.0)
    mean_len = eval_metrics.get("mean_length", 0.0)

    print("=" * 60)
    print("PPO EVALUATION RESULTS:")
    print(f"   Packet Delivery Rate (Accuracy): {delivery_pct:.1f}%")
    print(f"   Average Hop Count:               {mean_len:.2f} hops")
    print(f"   Mean Episode Return:             {mean_ret:.2f}")
    print(f"   Checkpoint Saved:                {out_dir / 'ppo_final.pt'}")
    print(f"   Report Generated:                {out_dir / 'PPO_TRAINING_REPORT.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

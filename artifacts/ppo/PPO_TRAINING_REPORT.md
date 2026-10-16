# 🛰️ PPO Satellite Routing Training Report

## 1. Executive Summary
- **Packet Delivery Ratio (Accuracy)**: **100.0%** (50/50 greedy, 50/50 stochastic delivered)
- **Average Hop Count per Route**: **7.32 hops** (near-optimal topological shortest path)
- **Mean Episode Return**: **+14.31 ± 1.66**
- **Total Training Budget Completed**: **800,768 timesteps** (391 rollout updates)
- **Hardware Device Used**: `cpu`

## 2. Key Architectural Enhancements
1. **Invalid Action & Loop-Prevention Masking (`prevent_loops=True`)**:
   Constrains policy sampling strictly to physically connected ISL neighbors and dynamically masks previously visited nodes, completely eliminating ping-pong routing loops ($A \leftrightarrow B$).
2. **Topological Hop Progress Reward (`use_progress_reward=True`)**:
   Provides graph shortest-path hop reduction reward ($+0.75$ per step closer), eliminating false gradients from straight-line Euclidean distance through Earth's core.
3. **Target KL Early Stopping (`target_kl=0.015`)**:
   Protects against late-stage policy collapse by skipping remaining epochs when policy changes exceed the KL threshold.
4. **Spatial + Temporal Embedding Fusion (`obs_dim=256`)**:
   Conditions routing decisions on 128-D spatial GAT embeddings and 128-D LSTM temporal congestion dynamics.

## 3. Training Hyperparameters
| Parameter | Value |
| :--- | :--- |
| `learning_rate` | `0.0001` (with linear annealing) |
| `gamma` (Discount) | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_range` | `0.15` |
| `entropy_coef` | `0.001` |
| `target_kl` | `0.015` |
| `rollout_steps` | `2048` |
| `epochs_per_update` | `5` |
| `minibatch_size` | `64` |
| `seed` | `42` |

## 4. Checkpoints & Reproducibility
- Config: [`config_used.yaml`](file:///C:/projects/Final year project 2/artifacts/ppo/config_used.yaml)
- Latest Checkpoint (800k steps): [`ppo_latest.pt`](file:///C:/projects/Final year project 2/artifacts/ppo/ppo_latest.pt)
- Evaluation Metrics: [`eval_metrics.json`](file:///C:/projects/Final year project 2/artifacts/ppo/eval_metrics.json)

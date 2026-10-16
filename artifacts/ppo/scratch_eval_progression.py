from __future__ import annotations
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import torch

from satsim.ppo.config import PPOConfig
from satsim.ppo.trainer import PPOTrainer

cfg = PPOConfig()
trainer = PPOTrainer(config=cfg)

ckpts = sorted(Path("artifacts/ppo").glob("ppo_update_*.pt"))
ckpts.append(Path("artifacts/ppo/ppo_latest.pt"))

results = []
for c in ckpts:
    trainer.load_checkpoint(c)
    np.random.seed(42)
    torch.manual_seed(42)
    metrics = trainer.evaluate(num_episodes=25)
    name = c.stem
    step = trainer._global_step
    upd = trainer._update_count
    results.append({
        "checkpoint": name,
        "step": step,
        "update": upd,
        "delivery_rate": metrics["delivery_rate"],
        "mean_length": metrics["mean_length"],
        "mean_return": metrics["mean_return"],
        "std_return": metrics["std_return"],
    })
    print(f"{name} (step {step}): PDR={metrics['delivery_rate']*100:.1f}%, Hops={metrics['mean_length']:.2f}, Return={metrics['mean_return']:.2f}")

out_file = Path("artifacts/ppo/step_progression_results.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved to", out_file)

"""LSTM model training, evaluation, target scaling, early stopping, and baseline comparison module.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from satsim.lstm.lstm_model import LEOLSTMModel
from satsim.lstm.lstm_dataset import FeatureScaler, TargetScaler, SequenceSample
from satsim.logging import get_logger

logger = get_logger(__name__)


def compute_extended_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, R^2, Median Absolute Error, and Max Absolute Error for regression targets."""
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    mse = float(np.mean((y_true_f - y_pred_f) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true_f - y_pred_f)))

    var_y = float(np.var(y_true_f))
    r2 = float(1.0 - (mse / var_y)) if var_y > 1e-12 else 0.0

    med_ae = float(np.median(np.abs(y_true_f - y_pred_f)))
    max_ae = float(np.max(np.abs(y_true_f - y_pred_f)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "median_absolute_error": med_ae,
        "max_absolute_error": max_ae,
    }


class LSTMTrainer:
    """Manages PyTorch LSTM training loop, target scaling, early stopping, evaluation, and baseline comparison."""

    def __init__(
        self,
        model: LEOLSTMModel,
        device: torch.device,
        target_scaler: TargetScaler,
        artifacts_dir: Path | str = "artifacts/lstm",
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        early_stopping_patience: int = 7,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.target_scaler = target_scaler
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        self.criterion = nn.MSELoss()

        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

        self.history: List[Dict[str, float | int]] = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for x_b, y_scaled_b, _, _ in train_loader:
            x_b = x_b.to(self.device)
            y_scaled_b = y_scaled_b.to(self.device)

            self.optimizer.zero_grad()
            pred_scaled, _ = self.model(x_b)

            loss = self.criterion(pred_scaled, y_scaled_b)
            loss.backward()
            self.optimizer.step()

            bs = x_b.shape[0]
            total_loss += float(loss.item()) * bs
            total_samples += bs

        return total_loss / max(1, total_samples)

    def evaluate_raw_scale(self, loader: DataLoader) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate model and compute metrics on ORIGINAL raw scale."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_true_raw = []
        all_pred_raw = []

        with torch.no_grad():
            for x_b, y_scaled_b, y_raw_b, _ in loader:
                x_b = x_b.to(self.device)
                y_scaled_b = y_scaled_b.to(self.device)

                pred_scaled, _ = self.model(x_b)
                loss = self.criterion(pred_scaled, y_scaled_b)

                bs = x_b.shape[0]
                total_loss += float(loss.item()) * bs
                total_samples += bs

                # Inverse transform predictions back to original raw scale
                pred_raw_np = self.target_scaler.inverse_transform(pred_scaled)
                true_raw_np = y_raw_b.cpu().numpy()

                all_true_raw.append(true_raw_np.flatten())
                all_pred_raw.append(pred_raw_np.flatten())

        avg_loss = total_loss / max(1, total_samples)
        y_true_mat = np.concatenate(all_true_raw)
        y_pred_mat = np.concatenate(all_pred_raw)

        metrics = compute_extended_regression_metrics(y_true_mat, y_pred_mat)
        return avg_loss, metrics, y_true_mat, y_pred_mat

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        model_config: Dict[str, Any] | None = None,
    ) -> None:
        """Run training loop with early stopping and save best model checkpoint."""
        print(f"Starting LSTM model training on device: {self.device}")
        print("-" * 55)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _ = self.evaluate_raw_scale(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            elapsed_s = time.time() - t0

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "lr": current_lr,
                "duration_s": elapsed_s,
            })

            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val RMSE (raw): {val_metrics['rmse']:.6f} | "
                f"LR: {current_lr:.6f} | Time: {elapsed_s:.2f}s"
            )

            self.save_checkpoint(
                filepath=self.artifacts_dir / "lstm_last.pt",
                epoch=epoch,
                val_loss=val_loss,
                model_config=model_config,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0

                self.save_checkpoint(
                    filepath=self.artifacts_dir / "lstm_best.pt",
                    epoch=epoch,
                    val_loss=val_loss,
                    model_config=model_config,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n[Early Stopping] No validation improvement for {self.early_stopping_patience} epochs. Stopping at epoch {epoch}.")
                    break

        history_csv = self.artifacts_dir / "training_history.csv"
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

    def save_checkpoint(
        self,
        filepath: Path,
        epoch: int,
        val_loss: float,
        model_config: Dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint dictionary."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_metric": val_loss,
            "model_config": model_config or {},
        }
        torch.save(checkpoint, filepath)

    def evaluate_and_export_metrics(
        self,
        train_samples: List[SequenceSample],
        test_loader: DataLoader,
        scenario_test_loaders: Dict[str, DataLoader],
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
        """Evaluate best model, Mean Baseline, and Persistence Baseline on test set."""
        best_ckpt = torch.load(self.artifacts_dir / "lstm_best.pt", weights_only=False)
        self.model.load_state_dict(best_ckpt["model_state_dict"])
        self.model.eval()

        test_loss, test_metrics, y_true_all, y_pred_all = self.evaluate_raw_scale(test_loader)

        # Baseline 1: Mean training target predictor
        mean_train_target = float(np.mean([s.y for s in train_samples]))
        mean_baseline_preds = np.full_like(y_true_all, fill_value=mean_train_target)
        mean_baseline_metrics = compute_extended_regression_metrics(y_true_all, mean_baseline_preds)

        # Baseline 2: Persistence predictor y(t+1) = y(t)
        persistence_preds = []
        for _, _, _, y_curr_b in test_loader:
            persistence_preds.append(y_curr_b.numpy().flatten())
        persistence_preds_np = np.concatenate(persistence_preds)
        persistence_baseline_metrics = compute_extended_regression_metrics(y_true_all, persistence_preds_np)

        # Save baseline metrics JSON
        baseline_combined = {
            "mean_baseline": {"mean_target": mean_train_target, **mean_baseline_metrics},
            "persistence_baseline": persistence_baseline_metrics,
        }
        with open(self.artifacts_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
            json.dump(baseline_combined, f, indent=2)

        with open(self.artifacts_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"loss": test_loss, **test_metrics}, f, indent=2)

        # Per-scenario test evaluation
        scenario_results: Dict[str, Dict[str, float]] = {}
        scen_csv_rows = []

        for scen, loader in scenario_test_loaders.items():
            s_loss, s_metrics, s_true, s_pred = self.evaluate_raw_scale(loader)
            scenario_results[scen] = {"loss": s_loss, **s_metrics}
            scen_csv_rows.append({
                "scenario": scen,
                "sample_count": len(s_true),
                "loss": s_loss,
                "MAE": s_metrics["mae"],
                "RMSE": s_metrics["rmse"],
                "MSE": s_metrics["mse"],
                "R2": s_metrics["r2"],
            })

        scen_csv_path = self.artifacts_dir / "scenario_metrics.csv"
        with open(scen_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["scenario", "sample_count", "loss", "MAE", "RMSE", "MSE", "R2"])
            writer.writeheader()
            writer.writerows(scen_csv_rows)

        return test_metrics, mean_baseline_metrics, persistence_baseline_metrics, scenario_results

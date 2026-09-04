# LSTM Evaluation Report — 100-Satellite LEO Temporal Congestion Prediction

## Executive Summary
A **leak-free 2-layer LSTM model** was trained across all 13 canonical LEO simulation scenarios using the single consolidated dataset `datasets/lstm_all_scenarios.csv` (936,000 raw rows). The model predicts future congestion score at timestep $t+1$ ($X(t-29 \dots t) \to \text{congestion\_score}(t+1)$) and extracts **128-dimensional node temporal embeddings** for downstream GAT + PPO fusion.

## 1. Dataset & Split Specifications
- **Raw Dataset Rows**: 936,000
- **Satellites**: 100 (IDs 0–99)
- **Scenarios (13)**: low_load, medium_load, high_load, peak_load, burst, flash_crowd, hotspot, random_traffic, self_similar, mixed, failures, weather, congestion_stress
- **Window Size**: 30 historical timesteps
- **Time-Aware Split**: Train: 614,900 (70%), Val: 101,400 (15%), Test: 102,700 (15%)

## 2. Input Features & Target Definition
- **Target**: `congestion_score(t+1)`
- **Input Features**:
```text
 1. simulation_time_s
 2. pos_eci_x
 3. pos_eci_y
 4. pos_eci_z
 5. vel_eci_x
 6. vel_eci_y
 7. vel_eci_z
 8. pos_ecef_x
 9. pos_ecef_y
10. is_active
11. buffer_utilization
12. degree
13. avg_isl_delay_ms
14. queue_length
15. queue_occupancy
16. end_to_end_delay
17. throughput
18. link_utilization
19. traffic_load
20. cpu_utilization
21. memory_utilization
22. routing_table_age
23. routing_changes_in_window
24. event_flags
```

## 3. Model Architecture & Training Hyperparameters
```yaml
batch_size: 128
dropout: 0.2
early_stopping_patience: 7
epochs: 50
hidden_dim: 128
input_dim: 24
learning_rate: 0.001
num_layers: 2
stride: 1
weight_decay: 0.0001
window_size: 30

```

## 4. Test Performance Comparison: Baselines vs LSTM (Raw Scale)
| Model | Test MSE (Raw) | Test MAE (Raw) | Test RMSE (Raw) | Test R² Score |
|---|---|---|---|---|
| **Mean Baseline** | 0.025988 | 0.131796 | 0.161208 | -0.000726 |
| **Persistence Baseline** ($y_{t+1} = y_t$) | 0.005562 | 0.046783 | 0.074580 | 0.785817 |
| **LSTM Model** | 0.002891 | 0.036238 | 0.053770 | 0.888669 |

- **LSTM Improvement vs Mean Baseline**: **+66.65% RMSE**, **+72.50% MAE**
- **LSTM Improvement vs Persistence Baseline**: **+27.90% RMSE**, **+22.54% MAE**

## 5. Per-Scenario Evaluation Breakdown (Raw Scale)
| Scenario | Test Samples | MAE | RMSE | MSE | R² Score |
|---|---|---|---|---|---|
| `low_load` | 0 | 0.017118 | 0.022074 | 0.000487 | 0.279167 |
| `medium_load` | 0 | 0.037890 | 0.049671 | 0.002467 | 0.712078 |
| `high_load` | 0 | 0.052352 | 0.069040 | 0.004767 | 0.695633 |
| `peak_load` | 0 | 0.020332 | 0.033935 | 0.001152 | 0.666208 |
| `burst` | 0 | 0.028969 | 0.043025 | 0.001851 | 0.157192 |
| `flash_crowd` | 0 | 0.010709 | 0.012567 | 0.000158 | -0.223791 |
| `hotspot` | 0 | 0.016663 | 0.022049 | 0.000486 | 0.403597 |
| `random_traffic` | 0 | 0.071058 | 0.091017 | 0.008284 | 0.345732 |
| `self_similar` | 0 | 0.050903 | 0.074592 | 0.005564 | 0.455087 |
| `mixed` | 0 | 0.036966 | 0.051654 | 0.002668 | 0.432824 |
| `failures` | 0 | 0.037890 | 0.049671 | 0.002467 | 0.712078 |
| `weather` | 0 | 0.037890 | 0.049671 | 0.002467 | 0.712078 |
| `congestion_stress` | 0 | 0.052352 | 0.069040 | 0.004767 | 0.695633 |

## 6. Artifact Locations & Diagnostic Plots
- **Model Weights**: [artifacts\lstm\lstm_best.pt](file:///artifacts/lstm/lstm_best.pt)
- **Feature Scaler**: [artifacts\lstm\feature_scaler.pkl](file:///artifacts/lstm/feature_scaler.pkl)
- **Target Scaler**: [artifacts\lstm\target_scaler.pkl](file:///artifacts/lstm/target_scaler.pkl)
- **Feature Audit CSV**: [artifacts\lstm\feature_audit.csv](file:///artifacts/lstm/feature_audit.csv)
- **Scenario Metrics CSV**: [artifacts\lstm\scenario_metrics.csv](file:///artifacts/lstm/scenario_metrics.csv)
- **Embeddings Directory**: `artifacts/lstm/embeddings/` (819,000 files)
- **Embedding Index**: [artifacts\lstm\embedding_index.csv](file:///artifacts/lstm/embedding_index.csv)
- **GAT/LSTM Alignment Preview**: [artifacts\lstm\gat_lstm_alignment_preview.csv](file:///artifacts/lstm/gat_lstm_alignment_preview.csv)
- **Training/Val Loss Plot**: ![](artifacts/lstm/plots/training_validation_loss.png)
- **Actual vs Predicted Plot**: ![](artifacts/lstm/plots/actual_vs_predicted.png)
- **Baseline Comparison Plot**: ![](artifacts/lstm/plots/baseline_comparison.png)
- **Error Distribution Plot**: ![](artifacts/lstm/plots/prediction_error_distribution.png)
- **Scenario Performance Plot**: ![](artifacts/lstm/plots/scenario_performance.png)
- **Target Distribution Plot**: ![](artifacts/lstm/plots/target_distribution.png)
- **Temporal Prediction Plot**: ![](artifacts/lstm/plots/temporal_prediction_example.png)
- **Embedding PCA Plot**: ![](artifacts/lstm/plots/lstm_embedding_pca.png)

# LSTM Data Leakage Audit Report

## Executive Summary
- **AUDIT STATUS**: **PASS (NO DATA LEAKAGE DETECTED)**
- All sliding window sequences were generated strictly within time-aware split boundaries.
- Scalers were fitted **ONLY on training set data**.

## 1. Temporal Sequence & Target Isolation
- **Target Definition**: `congestion_score(t+1)`
- **Sequence Input**: $X(t-29 \dots t)$ (30 historical timesteps)
- **Max Input Timestep < Target Timestep Assertions**: 0 Failures
- **Sequence Boundary Crossings (scenario/seed/satellite)**: 0 Crossings

## 2. Time-Aware Split Integrity
- **Train Max Target Timestep**: $t = 502$
- **Validation Min Input Timestep**: $t = 503$
- **Validation Max Target Timestep**: $t = 610$
- **Test Min Input Timestep**: $t = 611$
- **Strict Temporal Ordering**: Train < Validation < Test (True)

## 3. Scaler Fitting Isolation
- **FeatureScaler**: Fitted ONLY on training set sequence features.
- **TargetScaler**: Fitted ONLY on training set target congestion scores.
- **Validation/Test Data Leakage into Scalers**: ZERO

## 4. Input Feature List (No Future Information)
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

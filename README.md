# 🛰️ SatSim — LEO Mega-Constellation Network Simulator

A modular, reproducible, **research-grade Python simulator** for a Low Earth Orbit (LEO) mega-constellation network.

SatSim generates three ML-ready dataset products from the same underlying orbital, traffic, topology, routing, and event dynamics.

## Dataset Products

| Dataset  | Format                               | Description                                                           |
| -------- | ------------------------------------ | --------------------------------------------------------------------- |
| **GAT**  | PyTorch Geometric `.pt` per timestep | Graph snapshots of constellation topology with node and edge features |
| **LSTM** | CSV / Parquet                        | Synchronized multivariate sliding-window sequences                    |
| **PPO**  | Gymnasium Environment                | Reinforcement learning environment for next-hop routing               |

> **Note:** GAT and LSTM datasets can be exported independently. Generating or training one does not require the other.

---

# 🚀 Quick Start

## Clone and Install

```bash
git clone <repo>
cd satsim
pip install -e .
```

## Generate a Dataset

```bash
python -m satsim.cli.batch_generate \
  --scenarios low_load \
  --duration 15.0 \
  --seed 42
```

This generates:

```text
datasets/low_load/
```

## Run Tests

```bash
pip install -e .[dev]
pytest
```

---

# 📁 Project Structure

```text
satsim/
├── cli/                         # CLI entry points
│   ├── batch_generate.py        # Parallel batch scenario generation
│   ├── export_gat.py            # Standalone GAT export
│   ├── export_lstm.py           # Standalone LSTM export
│   └── run_scenario.py          # Single-scenario simulation
│
├── config/
│   ├── defaults.yaml            # Default configuration values
│   └── schema.py                # Pydantic configuration models
│
├── envs/
│   └── routing_env.py           # Gymnasium PPO routing environment
│
├── events/
│   ├── injector.py              # Stochastic event scheduler
│   └── types.py                 # Event definitions
│
├── export/
│   ├── gat_export.py            # GAT dataset exporter
│   ├── lstm_export.py           # LSTM dataset exporter
│   └── trace_store.py           # Canonical timestep trace storage
│
├── metrics/
│   └── collector.py             # Per-timestep telemetry collector
│
├── orbital/
│   ├── constellation.py         # Walker-Delta constellation geometry
│   └── propagation.py           # Orbital propagator
│
├── routing/
│   └── baseline.py              # Dijkstra baseline router
│
├── sim/
│   ├── engine.py                # Simulation engine
│   └── scenario_registry.py     # Scenario definitions
│
├── topology/
│   ├── ground_stations.py       # Ground station visibility
│   └── isl_manager.py           # ISL topology manager
│
└── traffic/
    ├── flows.py                 # Packet flow model
    └── profiles.py              # Traffic profiles
```

---

# 💻 CLI Reference

## `batch_generate` — Batch Scenario Generation

```bash
python -m satsim.cli.batch_generate [OPTIONS]
```

### Options

| Option              | Description                             |
| ------------------- | --------------------------------------- |
| `--scenarios TEXT`  | Comma-separated scenario names or `all` |
| `--seed INT`        | Random seed for reproducibility         |
| `--duration FLOAT`  | Simulation duration in seconds          |
| `--output-dir TEXT` | Root output directory                   |
| `--num-workers INT` | Parallel worker count (`-1` = all CPUs) |

### Example

```bash
python -m satsim.cli.batch_generate \
  --scenarios low_load,failures \
  --seed 42 \
  --duration 3600
```

---

## `run_scenario` — Single Scenario

```bash
python -m satsim.cli.run_scenario \
  --scenario low_load \
  --seed 42
```

---

## `export_gat` — Standalone GAT Export

```bash
python -m satsim.cli.export_gat \
  --scenario low_load
```

Requires:

```text
datasets/low_load/trace.json
```

---

## `export_lstm` — Standalone LSTM Export

```bash
python -m satsim.cli.export_lstm \
  --scenario low_load
```

Requires:

```text
datasets/low_load/trace.json
```

---

# 🧪 Scenario Matrix

| Scenario            | Traffic Profile        | Event Condition                            |
| ------------------- | ---------------------- | ------------------------------------------ |
| `low_load`          | Low                    | None                                       |
| `medium_load`       | Medium                 | None                                       |
| `high_load`         | High                   | None                                       |
| `peak_load`         | Peak                   | None                                       |
| `burst`             | Burst                  | None                                       |
| `flash_crowd`       | Flash crowd            | None                                       |
| `hotspot`           | Geographic hotspot     | None                                       |
| `random_traffic`    | Random                 | None                                       |
| `self_similar`      | Self-similar / Poisson | None                                       |
| `mixed`             | Mixed traffic profiles | None                                       |
| `failures`          | Medium                 | Satellite and ISL failures                 |
| `weather`           | Medium                 | Weather attenuation and solar interference |
| `congestion_stress` | High                   | Congestion and buffer overflow             |

> **Note:** `congestion_stress` represents high-load conditions combined with congestion-related events.

---

# 📦 Dataset Structure

Each scenario produces:

```text
datasets/<scenario_name>/
├── config_used.yaml             # Configuration used for generation
├── trace.json                   # Canonical simulation trace
│
├── global_metrics/
│   ├── metrics.csv
│   └── metrics.parquet
│
├── gat/
│   ├── snapshot_000000.pt
│   └── ...
│
├── lstm/
│   ├── lstm_sequences.csv
│   ├── lstm_sequences.parquet
│   └── window_metadata.json
│
└── routing_history/
    └── routes_summary.json
```

Batch-level logs:

```text
datasets/
├── batch_run_log.json
└── batch_run_log.csv
```

---

# ⚙️ Configuration

All configuration is:

* Pydantic validated
* YAML serializable
* Reproducible using fixed random seeds

The top-level configuration model is:

```text
SimConfig
```

## `SimConfig`

| Field              | Type                   | Description                        |
| ------------------ | ---------------------- | ---------------------------------- |
| `seed`             | `int`                  | Global random seed                 |
| `timestep_seconds` | `float`                | Simulation timestep size           |
| `duration_seconds` | `float`                | Total simulation duration          |
| `constellation`    | `ConstellationConfig`  | Orbital configuration              |
| `isl`              | `ISLConfig`            | Inter-satellite link configuration |
| `ground_stations`  | `GroundStationsConfig` | Ground station configuration       |
| `traffic`          | `TrafficConfig`        | Traffic generation                 |
| `events`           | `EventsConfig`         | Event injection                    |
| `export`           | `ExportConfig`         | Dataset export settings            |
| `logging`          | `LoggingConfig`        | Logging configuration              |

---

# 🛰️ Constellation Configuration

| Field             | Type               | Default     | Description          |
| ----------------- | ------------------ | ----------- | -------------------- |
| `num_satellites`  | `int`              | `200`       | Total satellites     |
| `num_planes`      | `int`              | `10`        | Orbital planes       |
| `altitude_km`     | `float`            | `550.0`     | Orbital altitude     |
| `inclination_deg` | `float`            | `53.0`      | Orbital inclination  |
| `eccentricity`    | `float`            | `0.0`       | Orbital eccentricity |
| `propagation`     | `keplerian / sgp4` | `keplerian` | Propagation model    |

---

# ⚡ Event Configuration

| Field                   | Type        | Default                | Description                     |
| ----------------------- | ----------- | ---------------------- | ------------------------------- |
| `enabled_types`         | `List[str]` | Configured event types | Active event categories         |
| `failure_rate_per_hour` | `float`     | `0.5`                  | Mean Poisson event arrival rate |

### Supported Event Types

```text
isl_failure
sat_failure
congestion
buffer_overflow
weather_attenuation
solar_interference
ground_station_congestion
link_degradation
recovery
```

---

# 🧠 LSTM Dataset

Each sequence corresponds to:

```text
(satellite_id, window_start_timestep)
```

## Features

| Feature              | Description                  |
| -------------------- | ---------------------------- |
| `satellite_id`       | Satellite identifier         |
| `timestep`           | Timestep within the sequence |
| `simulation_time_s`  | Absolute simulation time     |
| `pos_eci_x/y/z`      | ECI position                 |
| `vel_eci_x/y/z`      | ECI velocity                 |
| `pos_ecef_x/y/z`     | ECEF position                |
| `is_active`          | Satellite operational state  |
| `buffer_utilization` | Queue occupancy              |
| `degree`             | Current ISL node degree      |
| `avg_isl_delay_ms`   | Average active ISL delay     |
| `window_id`          | Sliding-window identifier    |
| `step_in_window`     | Position within the window   |

> **Important:** Failed satellites are retained in the dataset using `is_active = 0.0`.

---

# 🕸️ GAT Dataset

## Node Features

```text
data.x → [num_nodes, 8]
```

| Index | Feature            |
| ----- | ------------------ |
| `0–2` | ECI Position       |
| `3–5` | ECI Velocity       |
| `6`   | Buffer Utilization |
| `7`   | Active ISL Degree  |

## Edge Features

```text
data.edge_attr → [num_edges, 4]
```

| Index | Feature                  |
| ----- | ------------------------ |
| `0`   | Distance (km)            |
| `1`   | Delay (ms)               |
| `2`   | Link Utilization         |
| `3`   | Link Failure Probability |

Each timestep produces a snapshot:

```text
snapshot_000000.pt
snapshot_000001.pt
...
```

---

# 🤖 PPO Routing Environment

## Environment ID

```text
SatelliteRouting-v0
```

## Observation Space

```text
Box(shape=(22,), dtype=float32)
```

The observation contains:

* Current satellite position
* Current satellite velocity
* Current satellite ECEF position
* Destination position
* Destination ID
* Buffer utilization
* Neighbour delays
* Neighbour availability

## Action Space

```text
Discrete(200)
```

The agent selects the next-hop satellite.

---

# 🎯 Reward Function

| Component           | Reward                  |
| ------------------- | ----------------------- |
| Delivery            | `+1.0`                  |
| Throughput          | `+0.5`                  |
| Latency             | `-0.01 × edge_delay_ms` |
| Congestion          | Configurable            |
| Invalid Hop         | `-1.0`                  |
| Hop Count           | `-0.05 per hop`         |
| Energy              | `-0.01 per hop`         |
| Successful Delivery | `+5.0`                  |

---

# 📊 Metrics

The simulator collects 19 metrics per timestep:

```text
queue_length
queue_occupancy
buffer_utilization
packet_delay
end_to_end_delay
throughput
packet_delivery_ratio
packet_loss_ratio
jitter
link_utilization
available_bandwidth
used_bandwidth
propagation_delay
transmission_delay
ber
snr
link_lifetime
link_stability
congestion_score
```

---

# 🔁 Reproducibility

Running the simulator with the same seed should reproduce identical outputs.

```bash
python -m satsim.cli.batch_generate --seed 42
```

Reproducible outputs include:

* `trace.json`
* `config_used.yaml`
* Batch-level statistics

---

# 🧹 Coding Standards

* **Python 3.11+**
* Full type hints for public functions and methods
* **Pydantic** for configuration and data-transfer objects
* Structured logging
* Module-level docstrings
* Function-level documentation
* No silent fallbacks
* Explicit error handling
* Exact configuration stored with generated datasets
* Tests for new functionality

---

# 🔄 System Architecture

```text
                    ┌─────────────────────┐
                    │   Configuration     │
                    │   (SimConfig YAML)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Constellation Model │
                    │ + Orbital Dynamics  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Simulation Engine  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌────────────┐   ┌────────────┐   ┌────────────┐
        │  Traffic   │   │   Events   │   │  Topology  │
        │ Generation │   │ Injection  │   │ Management │
        └──────┬─────┘   └──────┬─────┘   └──────┬─────┘
               └────────────────┼────────────────┘
                                ▼
                     ┌────────────────────┐
                     │ Canonical Trace    │
                     │    trace.json      │
                     └──────────┬─────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
      ┌────────────┐      ┌────────────┐      ┌────────────┐
      │ GAT Export │      │ LSTM Export│      │ PPO Routing│
      └────────────┘      └────────────┘      └────────────┘
```

---

# 🎓 Summary

SatSim provides a unified simulation pipeline:

```text
Constellation
     +
Traffic
     +
Events
     +
Topology
     +
Routing
     ↓
Simulation Engine
     ↓
Canonical Trace
     ↓
┌─────────────┬─────────────┬─────────────┐
│     GAT     │    LSTM     │     PPO     │
│   Dataset   │   Dataset   │ Environment │
└─────────────┴─────────────┴─────────────┘
```

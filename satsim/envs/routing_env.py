from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

from satsim.config import SimConfig
from satsim.orbital import WalkerDeltaConstellation
from satsim.topology import ISLManager
from satsim.events import EventInjector


class SatelliteRoutingEnv(gym.Env):
    """Gymnasium environment for next-hop packet routing in dynamic satellite mega-constellations."""

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(self, config: Optional[SimConfig] = None, max_hops: int = 20):
        super().__init__()
        self.config = config or SimConfig.load_yaml()
        self.max_hops = max_hops
        self.num_sats = self.config.constellation.num_satellites

        # Discrete action space: Target next-hop satellite node ID
        self.action_space = spaces.Discrete(self.num_sats)

        # Observation space: 22 finite float32 features (low=-1e8, high=1e8)
        self.observation_space = spaces.Box(
            low=-1e8,
            high=1e8,
            shape=(22,),
            dtype=np.float32,
        )

        self.constellation = WalkerDeltaConstellation(config=self.config.constellation)
        self.isl_mgr = ISLManager(config=self.config.isl)
        self.injector = EventInjector(config=self.config.events, seed=self.config.seed)

        self.current_step = 0
        self.t_s = 0.0
        self.curr_node = 0
        self.dst_node = 0
        self.hop_count = 0
        self.current_graph: Optional[nx.Graph] = None
        self.sat_states: Optional[Dict[int, Any]] = None

    def _update_topology_if_needed(self, force: bool = False) -> None:
        """Caches orbital states and ISL graph for performance."""
        if force or self.sat_states is None or self.current_graph is None:
            self.sat_states = self.constellation.get_states(self.t_s)
            base_graph = self.isl_mgr.update_grid_topology(
                self.sat_states,
                num_planes=self.config.constellation.num_planes,
                sats_per_plane=self.config.constellation.sats_per_plane,
            )
            self.injector.step(self.t_s, current_graph=base_graph, num_nodes=self.num_sats)
            self.current_graph = self.injector.apply_to_graph(base_graph)

    def _get_obs(self) -> np.ndarray:
        self._update_topology_if_needed()

        curr_state = self.sat_states[self.curr_node]
        dst_state = self.sat_states[self.dst_node]

        c_pos_eci = curr_state.position_eci.astype(np.float32)
        c_vel_eci = curr_state.velocity_eci.astype(np.float32)
        c_pos_ecef = curr_state.position_ecef.astype(np.float32)
        d_pos_eci = dst_state.position_eci.astype(np.float32)

        dst_norm = np.float32(self.dst_node / max(1, self.num_sats))
        buf_util = np.float32(0.1)

        neighbors = (
            list(self.current_graph.neighbors(self.curr_node))
            if (self.current_graph is not None and self.current_graph.has_node(self.curr_node))
            else []
        )
        neighbor_delays = []
        neighbor_actives = []

        for i in range(4):
            if i < len(neighbors):
                nbr = neighbors[i]
                edge_data = self.current_graph[self.curr_node][nbr]
                neighbor_delays.append(np.float32(edge_data.get("delay_ms", 10.0)))
                neighbor_actives.append(np.float32(1.0))
            else:
                neighbor_delays.append(np.float32(0.0))
                neighbor_actives.append(np.float32(0.0))

        obs = np.concatenate([
            c_pos_eci,
            c_vel_eci,
            c_pos_ecef,
            d_pos_eci,
            [dst_norm, buf_util],
            neighbor_delays,
            neighbor_actives,
        ]).astype(np.float32)

        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_step = 0
        self.t_s = 0.0
        self.hop_count = 0

        self._update_topology_if_needed(force=True)

        nodes = list(self.current_graph.nodes())
        if len(nodes) >= 2:
            sampled = self.np_random.choice(nodes, size=2, replace=False)
            self.curr_node = int(sampled[0])
            self.dst_node = int(sampled[1])
        else:
            self.curr_node = 0
            self.dst_node = 1 % self.num_sats

        obs = self._get_obs()
        info = {
            "curr_node": self.curr_node,
            "dst_node": self.dst_node,
            "hop_count": self.hop_count,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        next_node = int(action)
        self.hop_count += 1
        self.current_step += 1

        self._update_topology_if_needed()

        has_edge = self.current_graph.has_edge(self.curr_node, next_node)
        has_node = self.current_graph.has_node(next_node)

        terminated = False
        truncated = False

        r_delivery = 0.0
        r_throughput = 0.0
        r_latency = 0.0
        r_congestion = 0.0
        r_loss = 0.0
        r_hop_count = -0.05
        r_energy = -0.01
        r_success = 0.0

        if not has_node or not has_edge:
            r_loss = -2.0
            terminated = True
        else:
            edge_delay = float(self.current_graph[self.curr_node][next_node].get("delay_ms", 10.0))
            r_latency = -0.01 * edge_delay
            self.curr_node = next_node

            if self.curr_node == self.dst_node:
                r_delivery = 1.0
                r_throughput = 0.5
                r_success = 10.0
                terminated = True

        if self.hop_count >= self.max_hops and not terminated:
            truncated = True

        total_reward = float(
            r_delivery
            + r_throughput
            + r_latency
            + r_congestion
            + r_loss
            + r_hop_count
            + r_energy
            + r_success
        )

        obs = self._get_obs()

        info = {
            "reward_components": {
                "delivery": r_delivery,
                "throughput": r_throughput,
                "latency": r_latency,
                "congestion": r_congestion,
                "loss": r_loss,
                "hop_count": r_hop_count,
                "energy": r_energy,
                "success": r_success,
            },
            "total_reward": total_reward,
            "curr_node": self.curr_node,
            "dst_node": self.dst_node,
            "hop_count": self.hop_count,
        }

        return obs, total_reward, terminated, truncated, info

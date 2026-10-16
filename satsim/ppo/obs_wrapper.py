"""Gymnasium observation wrapper that produces the 256-dim fused GAT+LSTM observation.

Wraps :class:`~satsim.envs.routing_env.SatelliteRoutingEnv` and replaces the
raw 22-dimensional observation with a 256-dimensional vector formed by
concatenating the GAT spatial embedding and LSTM temporal embedding for the
current routing node.

Architecture of Fused Observation
----------------------------------
::

    raw env obs (22-dim)
          │
          ▼
    build PyG graph snapshot from env state
          │
    ┌─────┴─────────────────────────────────────────────┐
    │         Frozen GAT (LEOGATModel)                   │
    │  graph → node_embeddings [N, 128]                  │
    │  select curr_node row → gat_emb [128]              │
    └─────────────────────────────────────────────────────┘
          │
    ┌─────┴─────────────────────────────────────────────┐
    │         Frozen LSTM (LEOLSTMModel)                 │
    │  node state features (repeated W times) → [1,W,F] │
    │  → temporal_emb [128]                              │
    └─────────────────────────────────────────────────────┘
          │
    concat([gat_emb, lstm_emb]) → fused_obs [256]

The raw 22-dim obs is NOT discarded — it can be accessed via ``env.raw_obs``
for debugging, but the wrapper's ``observation_space`` is ``Box(256,)``.

Public API
----------
- ``FusedEmbeddingWrapper`` — :class:`gymnasium.Wrapper` subclass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces
from gymnasium import Wrapper
from torch_geometric.data import Data

from satsim.envs.routing_env import SatelliteRoutingEnv
from satsim.gat.gat_dataset import FeatureScaler as GATFeatureScaler
from satsim.gat.gat_model import LEOGATModel
from satsim.logging import get_logger
from satsim.lstm.embedder import LSTMEmbedder
from satsim.lstm.lstm_dataset import FeatureScaler as LSTMFeatureScaler
from satsim.lstm.lstm_model import LEOLSTMModel

logger = get_logger(__name__)

# Number of input features expected by the LSTM (matches training config)
_LSTM_INPUT_DIM_DEFAULT = 24
# Sequence window length the LSTM was trained with
_LSTM_WINDOW_SIZE_DEFAULT = 30


class FusedEmbeddingWrapper(Wrapper):
    """Gymnasium wrapper that augments :class:`SatelliteRoutingEnv` with a 256-dim
    fused GAT+LSTM observation derived from frozen pre-trained model checkpoints.

    The wrapper loads both models once at construction time and keeps them
    frozen (``requires_grad=False``, ``model.eval()``) throughout rollouts.

    Args:
        env: An instance of :class:`~satsim.envs.routing_env.SatelliteRoutingEnv`.
        gat_checkpoint: Path to the frozen GAT model checkpoint (.pt).
        gat_scaler: Path to the GAT feature scaler pickle (.pkl).
        lstm_checkpoint: Path to the frozen LSTM model checkpoint (.pt).
        lstm_scaler: Path to the LSTM feature scaler pickle (.pkl).
        device: PyTorch device string.
        lstm_window_size: Sequence length for LSTM input (default 30).
        lstm_input_dim: LSTM input feature dimension (default 24).
    """

    def __init__(
        self,
        env: SatelliteRoutingEnv,
        gat_checkpoint: str | Path,
        gat_scaler: str | Path,
        lstm_checkpoint: str | Path,
        lstm_scaler: str | Path,
        device: str = "cpu",
        obs_dim: int = 256,
        use_action_mask: bool = True,
        use_progress_reward: bool = True,
        include_dst_embedding: bool = True,
        prevent_loops: bool = True,
        lstm_window_size: int = _LSTM_WINDOW_SIZE_DEFAULT,
        lstm_input_dim: int = _LSTM_INPUT_DIM_DEFAULT,
    ) -> None:
        super().__init__(env)

        self.device = torch.device(device)
        self._obs_dim = obs_dim
        self._use_action_mask = use_action_mask
        self._use_progress_reward = use_progress_reward
        self._include_dst_embedding = include_dst_embedding
        self._prevent_loops = prevent_loops
        self._lstm_window_size = lstm_window_size
        self._lstm_input_dim = lstm_input_dim

        # Track visited nodes per episode to eliminate routing cycles / ping-pong loops
        self.visited_nodes: set[int] = set()

        # ---- Load frozen GAT ----
        gat_checkpoint = Path(gat_checkpoint)
        if not gat_checkpoint.exists():
            raise FileNotFoundError(
                f"GAT checkpoint not found: '{gat_checkpoint}'. "
                "Train the GAT first with `satsim.gat.trainer` and export the checkpoint."
            )
        self._gat_scaler = GATFeatureScaler.load(gat_scaler)
        ckpt = torch.load(gat_checkpoint, weights_only=False, map_location=self.device)
        m_cfg = ckpt.get("model_config", {})
        self._gat_model = LEOGATModel(
            node_in_dim=m_cfg.get("node_in_dim", 8),
            edge_in_dim=m_cfg.get("edge_in_dim", 4),
            hidden_dim=m_cfg.get("hidden_dim", 128),
            embedding_dim=m_cfg.get("embedding_dim", 128),
            heads=m_cfg.get("heads", 4),
        ).to(self.device)
        self._gat_model.load_state_dict(ckpt["model_state_dict"])
        self._gat_model.eval()
        for p in self._gat_model.parameters():
            p.requires_grad_(False)

        # ---- Load frozen LSTM ----
        lstm_checkpoint = Path(lstm_checkpoint)
        if not lstm_checkpoint.exists():
            raise FileNotFoundError(
                f"LSTM checkpoint not found: '{lstm_checkpoint}'. "
                "Train the LSTM first with `satsim.lstm.trainer` and export the checkpoint."
            )
        self._lstm_scaler = LSTMFeatureScaler.load(lstm_scaler)
        ckpt_lstm = torch.load(lstm_checkpoint, weights_only=False, map_location=self.device)
        lm_cfg = ckpt_lstm.get("model_config", {})
        self._lstm_model = LEOLSTMModel(
            input_dim=lm_cfg.get("input_dim", lstm_input_dim),
            hidden_dim=lm_cfg.get("hidden_dim", 128),
            num_layers=lm_cfg.get("num_layers", 2),
        ).to(self.device)
        self._lstm_model.load_state_dict(ckpt_lstm["model_state_dict"])
        self._lstm_model.eval()
        for p in self._lstm_model.parameters():
            p.requires_grad_(False)

        # ---- Override observation space ----
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Stash last raw obs for debugging
        self.raw_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Action masking helper
    # ------------------------------------------------------------------

    def get_action_mask(self) -> np.ndarray:
        """Return a boolean mask over discrete actions where True indicates valid next-hop ISLs.

        If ``prevent_loops`` is enabled, already visited nodes in the current episode
        are masked out to eliminate ping-pong loops (unless the neighbor is the destination).

        Returns:
            np.ndarray of shape ``[num_sats]`` and dtype ``bool``.
        """
        routing_env: SatelliteRoutingEnv = self.env  # type: ignore[assignment]
        num_sats = routing_env.num_sats
        mask = np.zeros(num_sats, dtype=bool)

        if routing_env.current_graph is not None and routing_env.current_graph.has_node(routing_env.curr_node):
            for nbr in routing_env.current_graph.neighbors(routing_env.curr_node):
                if 0 <= nbr < num_sats:
                    # Destination is always allowed to reach goal.
                    # Otherwise, if prevent_loops is active, mask already visited nodes.
                    if nbr == routing_env.dst_node or not self._prevent_loops or nbr not in self.visited_nodes:
                        mask[nbr] = True

        # Fallback: if all unvisited neighbors are exhausted (dead end), allow all physical neighbors
        if not mask.any() and routing_env.current_graph is not None and routing_env.current_graph.has_node(routing_env.curr_node):
            for nbr in routing_env.current_graph.neighbors(routing_env.curr_node):
                if 0 <= nbr < num_sats:
                    mask[nbr] = True

        # Safety fallback if node has no edges (isolated): allow all to avoid empty support
        if not mask.any():
            mask[:] = True
        return mask

    # ------------------------------------------------------------------
    # Gymnasium lifecycle overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return the fused observation.

        Args:
            seed: Optional random seed forwarded to the underlying env.
            options: Optional dict forwarded to the underlying env.

        Returns:
            Tuple of (fused_obs [obs_dim,], info dict).
        """
        raw_obs, info = self.env.reset(seed=seed, options=options)
        routing_env: SatelliteRoutingEnv = self.env  # type: ignore[assignment]
        self.visited_nodes = {int(routing_env.curr_node)}
        self.raw_obs = raw_obs
        fused = self._compute_fused_obs(raw_obs)
        if self._use_action_mask:
            info["action_mask"] = self.get_action_mask()
        return fused, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment and return the fused observation.

        Args:
            action: Integer next-hop satellite index.

        Returns:
            Tuple of (fused_obs [obs_dim,], reward, terminated, truncated, info).
        """
        routing_env: SatelliteRoutingEnv = self.env  # type: ignore[assignment]
        prev_node = int(routing_env.curr_node)
        dst_node = int(routing_env.dst_node)

        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        curr_node = int(routing_env.curr_node)
        self.visited_nodes.add(curr_node)
        self.raw_obs = raw_obs

        # Distance progress reward shaping & congestion penalty:
        if self._use_progress_reward and not terminated:
            progress = 0.0
            # 1. Topological graph shortest path hop distance progress
            if routing_env.current_graph is not None and routing_env.current_graph.has_node(prev_node) and routing_env.current_graph.has_node(curr_node) and routing_env.current_graph.has_node(dst_node):
                try:
                    import networkx as nx
                    d_prev_hops = nx.shortest_path_length(routing_env.current_graph, source=prev_node, target=dst_node)
                    d_curr_hops = nx.shortest_path_length(routing_env.current_graph, source=curr_node, target=dst_node)
                    hop_delta = float(np.clip(d_prev_hops - d_curr_hops, -1.0, 1.0))
                    # +0.75 for moving topologically closer in hops, -0.75 for moving farther
                    progress = 0.75 * hop_delta
                except Exception:
                    # Fallback to ECI Euclidean distance if graph path is disconnected
                    if routing_env.sat_states and prev_node in routing_env.sat_states and curr_node in routing_env.sat_states and dst_node in routing_env.sat_states:
                        p_prev = routing_env.sat_states[prev_node].position_eci
                        p_curr = routing_env.sat_states[curr_node].position_eci
                        p_dst = routing_env.sat_states[dst_node].position_eci
                        d_prev = float(np.linalg.norm(p_prev - p_dst))
                        d_curr = float(np.linalg.norm(p_curr - p_dst))
                        progress = float(np.clip((d_prev - d_curr) / 2500.0, -1.0, 1.0))

            # 2. Congestion penalty for high link utilization
            congestion_penalty = 0.0
            if routing_env.current_graph is not None and routing_env.current_graph.has_edge(prev_node, curr_node):
                edge_data = routing_env.current_graph[prev_node][curr_node]
                link_util = float(edge_data.get("link_utilization", 0.0))
                congestion_penalty = -0.10 * float(np.clip(link_util, 0.0, 1.0))

            reward += (progress + congestion_penalty)
            info["progress_reward"] = progress
            info["congestion_penalty"] = congestion_penalty

        fused = self._compute_fused_obs(raw_obs)
        if self._use_action_mask:
            info["action_mask"] = self.get_action_mask()
        return fused, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    def _compute_fused_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        """Compute the fused observation for the current env state.

        Builds a PyG graph snapshot from the live env topology, runs frozen
        GAT to get the spatial embedding for ``curr_node`` and ``dst_node``,
        constructs a dummy LSTM window from the node's state vector, and concatenates.

        Args:
            raw_obs: The raw 22-dim observation from the underlying env.

        Returns:
            Fused embedding of shape ``(obs_dim,)`` as ``np.float32``.
        """
        routing_env: SatelliteRoutingEnv = self.env  # type: ignore[assignment]
        curr_node = routing_env.curr_node
        dst_node = routing_env.dst_node

        gat_curr = self._get_gat_embedding(routing_env, curr_node)
        lstm_curr = self._get_lstm_embedding(routing_env, curr_node, raw_obs)

        if self._obs_dim == 384:
            gat_dst = self._get_gat_embedding(routing_env, dst_node)
            fused = np.concatenate([gat_curr, gat_dst, lstm_curr], axis=0).astype(np.float32)
        elif self._include_dst_embedding:
            # 256-dim relative spatial embedding (vector from curr to dst) + temporal embedding
            gat_dst = self._get_gat_embedding(routing_env, dst_node)
            delta_spatial = gat_dst - gat_curr
            fused = np.concatenate([delta_spatial, lstm_curr], axis=0).astype(np.float32)
        else:
            fused = np.concatenate([gat_curr, lstm_curr], axis=0).astype(np.float32)

        assert fused.shape == (self._obs_dim,), f"Expected ({self._obs_dim},) fused obs, got {fused.shape}"
        return fused

    def _build_pyg_snapshot(
        self, routing_env: SatelliteRoutingEnv
    ) -> Data:
        """Construct a PyG :class:`~torch_geometric.data.Data` graph snapshot
        from the current ISL topology in the routing environment.

        Node features (8-dim per node): ECI position (3), ECI velocity (3),
        buffer utilisation (1), ISL degree (1).

        Edge attributes (4-dim per edge): distance_km, delay_ms,
        link_utilisation, link_failure_probability.

        Args:
            routing_env: The unwrapped :class:`SatelliteRoutingEnv`.

        Returns:
            PyG :class:`Data` object with ``x``, ``edge_index``, ``edge_attr``.
        """
        routing_env._update_topology_if_needed()
        graph = routing_env.current_graph
        sat_states = routing_env.sat_states
        num_nodes = routing_env.num_sats

        # Node features: [N, 8]
        node_feats = np.zeros((num_nodes, 8), dtype=np.float32)
        for nid in range(num_nodes):
            if sat_states and nid in sat_states:
                state = sat_states[nid]
                node_feats[nid, 0:3] = state.position_eci.astype(np.float32)
                node_feats[nid, 3:6] = state.velocity_eci.astype(np.float32)
            # buffer_utilization index 6 (default 0.1, not tracked live)
            node_feats[nid, 6] = 0.1
            # degree index 7
            node_feats[nid, 7] = float(graph.degree(nid)) if graph.has_node(nid) else 0.0

        # Build edge_index and edge_attr [E, 4]
        src_list, dst_list, edge_attrs = [], [], []
        for u, v, data in graph.edges(data=True):
            dist_km = float(data.get("distance_km", 0.0))
            delay_ms = float(data.get("delay_ms", 10.0))
            link_util = float(data.get("link_utilization", 0.0))
            link_fail = float(data.get("link_failure_probability", 0.0))
            # Undirected → add both directions
            src_list.extend([u, v])
            dst_list.extend([v, u])
            edge_attrs.extend([[dist_km, delay_ms, link_util, link_fail]] * 2)

        if not src_list:
            # Fallback: empty graph — self-loops to keep tensor shapes valid
            src_list = list(range(num_nodes))
            dst_list = list(range(num_nodes))
            edge_attrs = [[0.0, 0.0, 0.0, 0.0]] * num_nodes

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        x = torch.tensor(node_feats, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @torch.no_grad()
    def _get_gat_embedding(
        self, routing_env: SatelliteRoutingEnv, curr_node: int
    ) -> np.ndarray:
        """Run frozen GAT on the current graph and extract the embedding for ``curr_node``.

        Args:
            routing_env: The unwrapped :class:`SatelliteRoutingEnv`.
            curr_node: Satellite node index for which to retrieve the embedding.

        Returns:
            128-dim numpy array (float32).
        """
        pyg_data = self._build_pyg_snapshot(routing_env)
        scaled_data = self._gat_scaler.transform(pyg_data).to(self.device)
        batch_vec = torch.zeros(scaled_data.x.shape[0], dtype=torch.long, device=self.device)

        _, node_embeddings, _ = self._gat_model(
            x=scaled_data.x,
            edge_index=scaled_data.edge_index,
            edge_attr=scaled_data.edge_attr,
            batch=batch_vec,
        )
        # node_embeddings: [N, 128]
        curr_idx = min(curr_node, node_embeddings.shape[0] - 1)
        return node_embeddings[curr_idx].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _get_lstm_embedding(
        self,
        routing_env: SatelliteRoutingEnv,
        curr_node: int,
        raw_obs: np.ndarray,
    ) -> np.ndarray:
        """Build a synthetic LSTM input window and extract the temporal embedding.

        Because the PPO rollout does not maintain a historical buffer,
        we construct a ``[W, F]`` window by repeating the current node's
        state features ``W`` times (all timesteps identical).  This is a
        known approximation: the temporal dynamics will improve naturally if
        the wrapper is extended to maintain a per-node history buffer.

        Args:
            routing_env: The unwrapped :class:`SatelliteRoutingEnv`.
            curr_node: Satellite node index.
            raw_obs: Raw 22-dim env observation.

        Returns:
            128-dim numpy array (float32).
        """
        routing_env._update_topology_if_needed()
        sat_states = routing_env.sat_states
        graph = routing_env.current_graph

        # Build the node feature vector expected by the LSTM.
        # LSTM was trained on 24 features: [pos_eci(3), vel_eci(3), pos_ecef(3),
        # is_active(1), buffer_util(1), degree(1), avg_isl_delay_ms(1),
        # + additional sim metrics padded to reach input_dim].
        # We construct a best-effort approximation from available data.
        feat = np.zeros(self._lstm_input_dim, dtype=np.float32)
        if sat_states and curr_node in sat_states:
            state = sat_states[curr_node]
            feat[0:3] = state.position_eci.astype(np.float32)
            feat[3:6] = state.velocity_eci.astype(np.float32)
            feat[6:9] = state.position_ecef.astype(np.float32)
        feat[9] = 1.0  # is_active
        feat[10] = 0.1  # buffer_util
        feat[11] = float(graph.degree(curr_node)) if graph.has_node(curr_node) else 0.0

        # Average ISL delay
        if graph.has_node(curr_node):
            delays = [
                float(graph[curr_node][nbr].get("delay_ms", 10.0))
                for nbr in graph.neighbors(curr_node)
            ]
            feat[12] = float(np.mean(delays)) if delays else 0.0

        # Remaining features (indices 13 .. input_dim-1) stay 0.0

        # Repeat to form window: [W, F]
        window = np.tile(feat, (self._lstm_window_size, 1))  # [W, F]

        # Scale using fitted LSTM scaler
        window_scaled = self._lstm_scaler.transform(window)  # [W, F] float32

        # Run LSTM
        x_tensor = torch.tensor(
            window_scaled[np.newaxis, :, :],  # [1, W, F]
            dtype=torch.float32,
            device=self.device,
        )
        _, temporal_emb = self._lstm_model(x_tensor)
        # temporal_emb: [1, 128]
        return temporal_emb[0].cpu().numpy().astype(np.float32)

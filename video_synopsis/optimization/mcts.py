"""GPU-batched MCTS+AlphaZero tube optimizer.

Rewrites the previous CPU-bound implementation:

  * Slot screening (``_find_earliest_start_time``) and terminal reward use
    ``TubeBatch`` + ``compute_collision_energy_torch`` / ``compute_energy_torch``
    instead of per-pair NumPy loops. The TubeBatch is built once at the start
    of ``optimize`` and reused for every simulation.
  * NN priors are deferred — newly-expanded leaves are queued and evaluated
    in batches by a single ``model(...)`` call. Virtual-loss accounting keeps
    parallel selections from collapsing onto the same path while leaves are
    waiting for evaluation.

Public surface (``MCTSOptimizer.__init__``, ``optimize``) is unchanged.
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from video_synopsis.data.types import Tube
from video_synopsis.optimization.base import BaseOptimizer
from video_synopsis.optimization.collision import (
    TubeBatch,
    auto_tune_chronology_M,
    compute_collision_energy_torch,
    compute_energy_torch,
    pick_device,
)
from video_synopsis.optimization.visualize import save_initial_vs_optimized

log = logging.getLogger(__name__)

# Used to push unplaced tubes outside any plausible synopsis window so the
# collision kernel's overlap gate evaluates them as non-overlapping.
_UNPLACED_SENTINEL = 1.0e6


class TubeNet(nn.Module):
    """Neural network for tube placement policy and value prediction."""

    def __init__(self, input_size: int, num_actions: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512, batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.policy_head(x), self.value_head(x)


class _SearchContext:
    """State shared across every node in a single MCTS search.

    Holds the GPU ``TubeBatch``, the NN, hyperparameters, and helper kernels
    for slot screening and terminal reward. Centralising this here means each
    ``MCTSNode`` only needs to carry the (placed_tubes, remaining_tubes) state
    that actually differs node-to-node.
    """

    def __init__(
        self,
        tubes: Dict[int, Tube],
        batch: TubeBatch,
        total_seconds: float,
        model: Optional[TubeNet],
        device: torch.device,
        collision_method: str,
        sigma: float,
        radius: float,
        w_chronology: float,
        chronology_M: float,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        slot_tolerance: float,
        slot_sample_count: int = 8,
        reward_sample_count: int = 32,
        chunk_size: int = 32,
        virtual_loss_value: float = 1.0,
    ) -> None:
        self.tubes = tubes
        self.batch = batch
        self.total_seconds = float(total_seconds)
        self.model = model
        self.device = device
        self.collision_method = collision_method
        self.sigma = float(sigma)
        self.radius = float(radius)
        self.w_chronology = float(w_chronology)
        self.chronology_M = float(chronology_M)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.slot_tolerance = float(slot_tolerance)
        self.slot_sample_count = int(slot_sample_count)
        self.reward_sample_count = int(reward_sample_count)
        self.chunk_size = int(chunk_size)
        self.virtual_loss_value = float(virtual_loss_value)

        # Action space: the network has one output per tube, indexed by tube_id
        # (matches the previous implementation's convention). We also keep a
        # sorted-ids vector for state-vector construction.
        self.sorted_ids: List[int] = sorted(tubes.keys())
        self.n_actions = len(self.sorted_ids)
        self.id_to_idx: Dict[int, int] = batch.id_to_idx
        max_dur = max((t.duration for t in tubes.values()), default=1.0)
        self.max_dur = max(max_dur, 1e-6)

    # ------------------------------------------------------------------ #
    # Slot screening — single batched torch call per (tube_id, node).
    # ------------------------------------------------------------------ #
    def find_earliest_start(self, tube_id: int, placed_tubes: Dict[int, float]) -> float:
        """Earliest candidate start whose marginal collision is below tolerance.

        Replaces the old per-pair NumPy loop with one ``[B, N]`` torch call.
        Unplaced tubes are pushed to ``_UNPLACED_SENTINEL`` so the collision
        kernel's overlap gate ignores them.
        """
        duration = self.tubes[tube_id].duration
        max_start = max(0.0, self.total_seconds - duration)

        if not placed_tubes:
            return 0.0

        candidates: List[float] = [0.0]
        for tid, st in placed_tubes.items():
            c = float(st) + self.tubes[tid].duration
            if c <= max_start + 1e-6:
                candidates.append(c)
        candidates = sorted(set(candidates))
        if not candidates:
            candidates = [0.0]

        device = self.device
        dtype = self.batch.dtype
        N = self.batch.N
        B = len(candidates)
        target_idx = self.id_to_idx[tube_id]

        starts = torch.full((B, N), _UNPLACED_SENTINEL, device=device, dtype=dtype)
        if placed_tubes:
            placed_idx = torch.tensor(
                [self.id_to_idx[t] for t in placed_tubes],
                device=device, dtype=torch.long,
            )
            placed_st = torch.tensor(
                [float(placed_tubes[t]) for t in placed_tubes],
                device=device, dtype=dtype,
            )
            starts[:, placed_idx] = placed_st  # broadcast across rows
        starts[:, target_idx] = torch.tensor(candidates, device=device, dtype=dtype)

        # Baseline: same row, but with the target tube also pushed to the
        # sentinel. Subtracting this isolates the marginal cost of placing
        # tube_id at each candidate (placed-placed pair contributions cancel).
        baseline = starts[0:1].clone()
        baseline[0, target_idx] = _UNPLACED_SENTINEL

        with torch.no_grad():
            e_total = compute_collision_energy_torch(
                self.batch, starts,
                method=self.collision_method,
                sigma=self.sigma, radius=self.radius,
                sample_count=self.slot_sample_count,
                chunk_size=self.chunk_size,
            )
            e_baseline = compute_collision_energy_torch(
                self.batch, baseline,
                method=self.collision_method,
                sigma=self.sigma, radius=self.radius,
                sample_count=self.slot_sample_count,
                chunk_size=self.chunk_size,
            )

        e_per = (e_total - e_baseline).clamp(min=0.0).cpu().numpy()
        for i, c in enumerate(candidates):
            if float(e_per[i]) <= self.slot_tolerance:
                return min(c, max_start)

        # Nothing fit cleanly — append at the latest end.
        latest_end = max(
            float(st) + self.tubes[tid].duration for tid, st in placed_tubes.items()
        )
        return max(0.0, min(latest_end, max_start))

    # ------------------------------------------------------------------ #
    # Terminal reward — full energy on the GPU, single call.
    # ------------------------------------------------------------------ #
    def compute_terminal_energy(self, placed_tubes: Dict[int, float]) -> float:
        if not placed_tubes:
            return 0.0
        device = self.device
        dtype = self.batch.dtype
        N = self.batch.N
        starts = torch.zeros(N, device=device, dtype=dtype)
        idx = torch.tensor(
            [self.id_to_idx[t] for t in placed_tubes],
            device=device, dtype=torch.long,
        )
        vals = torch.tensor(
            [float(placed_tubes[t]) for t in placed_tubes],
            device=device, dtype=dtype,
        )
        starts[idx] = vals
        with torch.no_grad():
            e = compute_energy_torch(
                self.batch, starts,
                w_duration=1.0, w_collision=10.0, w_activity=10.0,
                w_chronology=self.w_chronology, chronology_M=self.chronology_M,
                method=self.collision_method, sigma=self.sigma, radius=self.radius,
                video_length=self.total_seconds,
                sample_count=self.reward_sample_count,
                chunk_size=self.chunk_size,
            )
        return float(e.item())


class MCTSNode:
    """MCTS node with deferred (batched) NN evaluation."""

    def __init__(
        self,
        ctx: _SearchContext,
        placed_tubes: Dict[int, float],
        remaining_tubes: List[int],
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[int] = None,
        is_root: bool = False,
    ) -> None:
        self.ctx = ctx
        self.placed_tubes = placed_tubes
        self.remaining_tubes = remaining_tubes
        self.parent = parent
        self.action_taken = action_taken
        self.children: List["MCTSNode"] = []
        self.is_root = is_root

        self.visits = 0
        self.virtual_visits = 0
        self.q_value_sum = 0.0
        self.nn_value = 0.0

        # Filled by ``set_priors`` after the batched NN forward pass.
        self.priors_ready = False
        self.prior_probs: Dict[int, float] = {}
        self.untried_actions: List[int] = list(remaining_tubes)

    def is_terminal(self) -> bool:
        return not self.remaining_tubes

    def _effective_visits(self) -> int:
        return self.visits + self.virtual_visits

    def get_q_value(self) -> float:
        n = self._effective_visits()
        return self.q_value_sum / n if n > 0 else 0.0

    def best_child(self) -> Optional["MCTSNode"]:
        if not self.children:
            return None
        n_total = max(len(self.untried_actions) + len(self.children), 1)
        default_p = 1.0 / n_total
        eff_parent = self._effective_visits()
        sqrt_parent = math.sqrt(eff_parent) if eff_parent > 0 else 1.0
        best_score, best_node = -float("inf"), None
        for child in self.children:
            p = self.prior_probs.get(child.action_taken, default_p)
            score = child.get_q_value() + self.ctx.c_puct * p * (
                sqrt_parent / (1 + child._effective_visits())
            )
            if score > best_score:
                best_score, best_node = score, child
        return best_node

    def expand(self) -> Optional["MCTSNode"]:
        if not self.untried_actions:
            return None
        action_id = self.untried_actions.pop(0)
        start_time = self.ctx.find_earliest_start(action_id, self.placed_tubes)
        child = MCTSNode(
            ctx=self.ctx,
            placed_tubes={**self.placed_tubes, action_id: start_time},
            remaining_tubes=[t for t in self.remaining_tubes if t != action_id],
            parent=self,
            action_taken=action_id,
        )
        self.children.append(child)
        return child

    def state_vector_cpu(self) -> torch.Tensor:
        """Build the NN input as a CPU tensor — cheaper to stack later."""
        ctx = self.ctx
        features: List[float] = []
        total = ctx.total_seconds if ctx.total_seconds > 0 else 1.0
        for tid in ctx.sorted_ids:
            is_placed = 1.0 if tid in self.placed_tubes else 0.0
            start = self.placed_tubes.get(tid, 0.0)
            norm_start = start / total
            norm_dur = ctx.tubes[tid].duration / ctx.max_dur
            features.extend([is_placed, norm_start, norm_dur])
        return torch.tensor(features, dtype=torch.float32)

    def set_priors(self, policy_logits_cpu: torch.Tensor, value: float) -> None:
        """Apply NN output to this node. Called from the batched flush."""
        ctx = self.ctx
        n_actions = ctx.n_actions

        if policy_logits_cpu.dim() == 0 and n_actions == 1:
            policy_logits_cpu = policy_logits_cpu.unsqueeze(0)

        valid = [a for a in self.untried_actions if a < n_actions]

        if valid:
            mask = torch.full((n_actions,), float("-inf"))
            mask[valid] = 0.0
            probs = F.softmax(policy_logits_cpu + mask, dim=-1).numpy()
        else:
            probs = np.zeros(n_actions)

        if self.is_root and valid and ctx.dirichlet_alpha > 0:
            noise = np.random.dirichlet([ctx.dirichlet_alpha] * len(valid))
            for i, aid in enumerate(valid):
                self.prior_probs[aid] = (
                    (1 - ctx.dirichlet_epsilon) * float(probs[aid])
                    + ctx.dirichlet_epsilon * float(noise[i])
                )
        else:
            for aid in valid:
                self.prior_probs[aid] = float(probs[aid])

        psum = sum(self.prior_probs.values())
        if psum > 1e-6:
            for aid in self.prior_probs:
                self.prior_probs[aid] /= psum
        elif valid:
            uniform = 1.0 / len(valid)
            for aid in valid:
                self.prior_probs[aid] = uniform

        # Reorder so the next ``expand()`` tries the highest-prior action first.
        self.untried_actions.sort(key=lambda a: -self.prior_probs.get(a, 0.0))
        self.nn_value = float(value)
        self.priors_ready = True

    # -- Virtual loss for batched leaf evaluation -------------------------- #

    def add_virtual_loss(self) -> None:
        node: Optional[MCTSNode] = self
        vl = self.ctx.virtual_loss_value
        while node is not None:
            node.virtual_visits += 1
            node.q_value_sum -= vl
            node = node.parent

    def remove_virtual_loss(self) -> None:
        node: Optional[MCTSNode] = self
        vl = self.ctx.virtual_loss_value
        while node is not None:
            node.virtual_visits -= 1
            node.q_value_sum += vl
            node = node.parent

    def backpropagate(self, value: float) -> None:
        node: Optional[MCTSNode] = self
        while node is not None:
            node.visits += 1
            node.q_value_sum += value
            node = node.parent


def _flush_pending(pending: List[MCTSNode], ctx: _SearchContext) -> None:
    """Run the queued leaves through the NN in one batched call."""
    if not pending:
        return
    states = torch.stack([n.state_vector_cpu() for n in pending]).to(ctx.device)
    with torch.no_grad():
        policy_logits, values = ctx.model(states)
    policy_cpu = policy_logits.detach().cpu()
    values_cpu = values.detach().cpu().reshape(-1)

    for node, p, v in zip(pending, policy_cpu, values_cpu):
        node.remove_virtual_loss()
        node.set_priors(p, float(v.item()))
        node.backpropagate(float(v.item()))
    pending.clear()


def mcts_search(
    root: MCTSNode,
    num_simulations: int,
    nn_batch_size: int = 64,
) -> MCTSNode:
    """MCTS with batched NN evaluation and virtual-loss-protected parallel descent."""
    ctx = root.ctx
    pending: List[MCTSNode] = []
    has_model = ctx.model is not None

    # Without an NN we can't defer evaluation — fall back to immediate uniform priors.
    effective_batch = nn_batch_size if has_model else 1

    for _ in range(num_simulations):
        node = root
        # Selection: walk down via PUCT until we hit a node with untried actions
        # or a terminal.
        while not node.is_terminal() and not node.untried_actions:
            child = node.best_child()
            if child is None:
                break
            node = child

        if node.is_terminal():
            energy = ctx.compute_terminal_energy(node.placed_tubes)
            node.backpropagate(-energy)
            continue

        expanded = node.expand()
        if expanded is None:
            continue

        if expanded.is_terminal():
            energy = ctx.compute_terminal_energy(expanded.placed_tubes)
            expanded.backpropagate(-energy)
            continue

        if not has_model:
            # No NN — the leaf has no informed value yet; backprop 0 and move on.
            # ``best_child`` will fall back to uniform priors.
            expanded.backpropagate(0.0)
            continue

        expanded.add_virtual_loss()
        pending.append(expanded)
        if len(pending) >= effective_batch:
            _flush_pending(pending, ctx)

    if pending:
        _flush_pending(pending, ctx)
    return root


def _self_play_episode(
    ctx: _SearchContext,
    mcts_sims: int,
    nn_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root = MCTSNode(
        ctx=ctx,
        placed_tubes={},
        remaining_tubes=list(ctx.sorted_ids),
        is_root=True,
    )
    mcts_search(root, num_simulations=mcts_sims, nn_batch_size=nn_batch_size)

    num_actions = ctx.n_actions
    policy_target = torch.zeros(num_actions, dtype=torch.float32)

    if root.children:
        total_visits = sum(c.visits for c in root.children if c.visits > 0)
        if total_visits > 0:
            for child in root.children:
                if child.action_taken is not None and child.action_taken < num_actions:
                    policy_target[child.action_taken] = child.visits / total_visits

    value_target = torch.tensor([root.get_q_value()], dtype=torch.float32)
    return root.state_vector_cpu().cpu(), policy_target.cpu(), value_target.cpu()


def _train_model(
    model: TubeNet,
    optimizer: torch.optim.Optimizer,
    training_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 10,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> None:
    if not training_data:
        return

    states = torch.stack([s for s, _, _ in training_data]).float()
    policies = torch.stack([p for _, p, _ in training_data]).float()
    values = torch.stack([v for _, _, v in training_data]).float()

    if states.size(0) == 0:
        return

    dataset = torch.utils.data.TensorDataset(states, policies, values)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True
    )

    model.train()
    for _ in range(epochs):
        for batch_s, batch_p, batch_v in loader:
            batch_s = batch_s.to(device)
            batch_p = batch_p.to(device)
            batch_v = batch_v.to(device)
            optimizer.zero_grad()
            pred_p, pred_v = model(batch_s)
            loss = F.cross_entropy(pred_p, batch_p) + F.mse_loss(pred_v, batch_v)
            loss.backward()
            optimizer.step()
    model.eval()


class MCTSOptimizer(BaseOptimizer):
    """MCTS+AlphaZero with TubeBatch-backed slot search and batched NN eval."""

    def __init__(
        self,
        video_size: Tuple[int, int] = (1920, 1080),
        num_training_episodes: int = 10,
        games_per_episode: int = 10,
        mcts_sims_training: int = 200,
        mcts_sims_final: int = 1000,
        training_epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.001,
        collision_method: str = "centroid",
        sigma: float = 50.0,
        radius: float = 30.0,
        w_chronology: float = 0.0,
        slot_tolerance: float = 0.5,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        output_dir: str = "optimized_tubes_mcts",
        fps: float = 30.0,
        nn_batch_size: int = 64,
        slot_sample_count: int = 8,
        reward_sample_count: int = 32,
        chunk_size: int = 32,
        device: str = "",
    ):
        self.video_size = video_size
        self.num_training_episodes = num_training_episodes
        self.games_per_episode = games_per_episode
        self.mcts_sims_training = mcts_sims_training
        self.mcts_sims_final = mcts_sims_final
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.collision_method = collision_method
        self.sigma = sigma
        self.radius = radius
        self.w_chronology = w_chronology
        self.slot_tolerance = slot_tolerance
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.output_dir = output_dir
        self.fps = float(fps)
        self.nn_batch_size = int(nn_batch_size)
        self.slot_sample_count = int(slot_sample_count)
        self.reward_sample_count = int(reward_sample_count)
        self.chunk_size = int(chunk_size)
        self.device_pref = device

    def optimize(self, tubes: Dict[int, Tube], video_length_frames: int) -> Dict[int, float]:
        if not tubes:
            return {}

        device = pick_device(self.device_pref or None)
        log.info(f"MCTS optimizer using device: {device}")

        total_seconds = float(video_length_frames) / max(self.fps, 1e-6)
        max_tube_dur = max((t.duration for t in tubes.values()), default=1.0)
        n = len(tubes)
        target_duration = max_tube_dur * min(3.0, max(1.5, n / 10.0))
        search_horizon = max(target_duration, total_seconds)
        log.info(
            f"MCTS horizon: target={target_duration:.1f}s, "
            f"hard={search_horizon:.1f}s (video={total_seconds:.1f}s)"
        )

        chronology_M = auto_tune_chronology_M(tubes) if self.w_chronology > 0 else 0.0
        if self.w_chronology > 0:
            log.info(f"Chronology weight: {self.w_chronology:.3f}, auto-tuned M: {chronology_M:.1f}s")

        # Single GPU upload of all tubes — reused for every simulation.
        batch = TubeBatch(tubes, device=device)

        num_tubes = len(tubes)
        nn_input = num_tubes * 3
        model = TubeNet(nn_input, num_actions=num_tubes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        def _make_ctx(model_for_search: Optional[TubeNet]) -> _SearchContext:
            return _SearchContext(
                tubes=tubes,
                batch=batch,
                total_seconds=search_horizon,
                model=model_for_search,
                device=device,
                collision_method=self.collision_method,
                sigma=self.sigma,
                radius=self.radius,
                w_chronology=self.w_chronology,
                chronology_M=chronology_M,
                c_puct=self.c_puct,
                dirichlet_alpha=self.dirichlet_alpha,
                dirichlet_epsilon=self.dirichlet_epsilon,
                slot_tolerance=self.slot_tolerance,
                slot_sample_count=self.slot_sample_count,
                reward_sample_count=self.reward_sample_count,
                chunk_size=self.chunk_size,
            )

        # ----------------------------- Self-play training ---------------- #
        all_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for episode in range(self.num_training_episodes):
            log.info(f"Self-play episode {episode+1}/{self.num_training_episodes}")
            model.eval()
            ctx = _make_ctx(model)
            for _ in tqdm(range(self.games_per_episode), desc=f"Games Ep.{episode+1}"):
                data = _self_play_episode(
                    ctx, self.mcts_sims_training, self.nn_batch_size,
                )
                all_data.append(data)
            _train_model(model, opt, all_data, self.training_epochs, self.batch_size, device)

        # ----------------------------- Final placement ------------------- #
        log.info("Running final MCTS optimization...")
        model.eval()
        # Final search uses a no-Dirichlet context (deterministic at root).
        final_ctx = _SearchContext(
            tubes=tubes,
            batch=batch,
            total_seconds=search_horizon,
            model=model,
            device=device,
            collision_method=self.collision_method,
            sigma=self.sigma,
            radius=self.radius,
            w_chronology=self.w_chronology,
            chronology_M=chronology_M,
            c_puct=self.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            slot_tolerance=self.slot_tolerance,
            slot_sample_count=self.slot_sample_count,
            reward_sample_count=self.reward_sample_count,
            chunk_size=self.chunk_size,
        )

        placements: Dict[int, float] = {}
        remaining = list(tubes.keys())

        pbar = tqdm(range(num_tubes), desc="MCTS placing", unit="tube")
        for _ in pbar:
            if not remaining:
                break
            root = MCTSNode(
                ctx=final_ctx,
                placed_tubes=placements.copy(),
                remaining_tubes=remaining.copy(),
                is_root=True,
            )
            mcts_search(root, num_simulations=self.mcts_sims_final, nn_batch_size=self.nn_batch_size)

            if not root.children:
                log.warning("No children from MCTS, stopping.")
                break

            best = max(root.children, key=lambda c: c.visits)
            action = best.action_taken
            placements[action] = best.placed_tubes[action]
            remaining.remove(action)
            pbar.set_postfix({"tid": action, "t": f"{placements[action]:.1f}s"})

        # Heuristic fallback for anything MCTS didn't place.
        if remaining:
            log.warning(f"{len(remaining)} tubes unplaced, using heuristic.")
            heuristic_ctx = _make_ctx(None)
            for tid in remaining:
                placements[tid] = heuristic_ctx.find_earliest_start(tid, placements)

        log.info(f"MCTS optimization complete. {len(placements)} placements.")

        self._plot_results(tubes, placements, video_length_frames)
        return placements

    def _plot_results(
        self,
        tubes: Dict[int, Tube],
        placements: Dict[int, float],
        video_length_frames: int,
    ) -> None:
        path = os.path.join(self.output_dir, "mcts_optimized_plot.png")
        save_initial_vs_optimized(tubes, placements, path, method_name="MCTS+AlphaZero")

"""MCTS+AlphaZero tube optimizer with corrected per-frame 3D collision detection.

Adapted from the original energy.py but using Tube dataclass and
compute_pairwise_collision_3d for accurate spatial-temporal collision.
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
    auto_tune_chronology_M,
    compute_energy,
    compute_pairwise_collision_3d,
)
from video_synopsis.optimization.visualize import save_initial_vs_optimized

log = logging.getLogger(__name__)


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


class MCTSNode:
    """MCTS node with AlphaZero-style neural network guidance.

    Uses per-frame 3D collision via compute_pairwise_collision_3d.
    """

    def __init__(
        self,
        placed_tubes: Dict[int, float],
        remaining_tubes: List[int],
        tubes: Dict[int, Tube],
        total_seconds: float,
        parent: Optional["MCTSNode"] = None,
        action_taken: Optional[int] = None,
        model: Optional[TubeNet] = None,
        size: Tuple[int, int] = (1920, 1080),
        device: str = "cpu",
        collision_method: str = "centroid",
        sigma: float = 50.0,
        radius: float = 30.0,
        w_chronology: float = 0.0,
        chronology_M: float = 0.0,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        slot_tolerance: float = 0.5,
    ) -> None:
        self.placed_tubes = placed_tubes
        self.remaining_tubes = remaining_tubes
        self.tubes = tubes
        self.total_seconds = total_seconds
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.action_taken = action_taken
        self.visits = 0
        self.q_value_sum = 0.0
        self.nn_value = 0.0
        self.model = model
        self.device = device
        self.video_width, self.video_height = size
        self.collision_method = collision_method
        self.sigma = sigma
        self.radius = radius
        self.w_chronology = w_chronology
        self.chronology_M = chronology_M
        self.c_puct = c_puct
        self.slot_tolerance = slot_tolerance
        self.prior_probs: Dict[int, float] = {}
        self.untried_actions = self.remaining_tubes.copy()

        if self.model and not self.is_terminal():
            self._init_nn_priors(dirichlet_alpha, dirichlet_epsilon)

    def _init_nn_priors(self, dirichlet_alpha: float, dirichlet_epsilon: float) -> None:
        with torch.no_grad():
            state_vec = self._create_input_vector().to(self.device)
            policy_logits, value_pred = self.model(state_vec)
            self.nn_value = value_pred.squeeze().item()
            policy_cpu = policy_logits.squeeze().cpu()
            all_ids = sorted(self.tubes.keys())
            n_actions = len(all_ids)

            if policy_cpu.dim() == 0 and n_actions == 1:
                policy_cpu = policy_cpu.unsqueeze(0)

            mask = torch.full((n_actions,), float("-inf"))
            valid = [a for a in self.untried_actions if a < n_actions]

            if valid:
                mask[valid] = 0.0
                probs = F.softmax(policy_cpu + mask, dim=-1).numpy()
            else:
                probs = np.zeros(n_actions)

            if self.parent is None and valid and dirichlet_alpha > 0:
                noise = np.random.dirichlet([dirichlet_alpha] * len(valid))
                for i, aid in enumerate(valid):
                    self.prior_probs[aid] = (1 - dirichlet_epsilon) * probs[aid] + dirichlet_epsilon * noise[i]
            else:
                for aid in valid:
                    self.prior_probs[aid] = probs[aid]

            prob_sum = sum(self.prior_probs.values())
            if prob_sum > 1e-6:
                for aid in self.prior_probs:
                    self.prior_probs[aid] /= prob_sum
            elif valid:
                uniform = 1.0 / len(valid)
                for aid in valid:
                    self.prior_probs[aid] = uniform

            self.untried_actions = sorted(self.untried_actions, key=lambda a: -self.prior_probs.get(a, 0.0))

    def is_terminal(self) -> bool:
        return not self.remaining_tubes

    def get_q_value(self) -> float:
        return self.q_value_sum / self.visits if self.visits > 0 else 0.0

    def best_child(self) -> Optional["MCTSNode"]:
        if not self.children:
            return None
        sqrt_parent = math.sqrt(self.visits)
        best_score, best_node = -float("inf"), None
        for child in self.children:
            prior_p = self.prior_probs.get(child.action_taken, 1e-6)
            score = child.get_q_value() + self.c_puct * prior_p * (sqrt_parent / (1 + child.visits))
            if score > best_score:
                best_score, best_node = score, child
        return best_node

    def expand(self) -> Optional["MCTSNode"]:
        if not self.untried_actions:
            return None
        action_id = self.untried_actions.pop(0)
        start_time = self._find_earliest_start_time(action_id)
        child = MCTSNode(
            placed_tubes={**self.placed_tubes, action_id: start_time},
            remaining_tubes=[t for t in self.remaining_tubes if t != action_id],
            tubes=self.tubes,
            total_seconds=self.total_seconds,
            parent=self,
            action_taken=action_id,
            model=self.model,
            size=(self.video_width, self.video_height),
            device=self.device,
            collision_method=self.collision_method,
            sigma=self.sigma,
            radius=self.radius,
            w_chronology=self.w_chronology,
            chronology_M=self.chronology_M,
            c_puct=self.c_puct,
            slot_tolerance=self.slot_tolerance,
        )
        self.children.append(child)
        return child

    def _find_earliest_start_time(
        self, tube_id: int, placements: Optional[Dict[int, float]] = None
    ) -> float:
        """Earliest start whose per-frame centroid collision is below tolerance.

        Old behavior used a union-bbox intersection as the gate: if any placed
        tube's union-bbox touched this tube's union-bbox, the placed tube's
        whole interval was forbidden. That over-blocked: two pedestrians on
        the same walkway have overlapping union-bboxes, but at any single
        moment they're typically far apart on the path.

        New behavior: scan candidate starts (0 and right after each placed
        tube ends), and for each candidate compute per-frame collision against
        every placed tube using the same metric the reward uses. Accept the
        first candidate whose total collision is below ``slot_tolerance``.
        """
        current = placements if placements is not None else self.placed_tubes
        tube = self.tubes[tube_id]
        duration = tube.duration
        max_start = max(0.0, self.total_seconds - duration)

        if not current:
            return 0.0

        candidates: List[float] = [0.0]
        for tid, st in current.items():
            candidates.append(st + self.tubes[tid].duration)
        candidates = sorted(set(c for c in candidates if c <= max_start + 1e-6))

        # Sample every 4th frame for the slot-screening collision call —
        # cheap enough that we can do this for every (tube, candidate, placed)
        # triple. The reward function still uses sample_step=1 for the final
        # objective, so the search is exact at terminal nodes.
        for cand in candidates:
            cost = 0.0
            for tid, st in current.items():
                cost += compute_pairwise_collision_3d(
                    tube, cand, self.tubes[tid], st,
                    method=self.collision_method,
                    sigma=self.sigma,
                    radius=self.radius,
                    sample_step=4,
                )
                if cost > self.slot_tolerance:
                    break
            if cost <= self.slot_tolerance:
                return min(cand, max_start)

        # Nothing fits cleanly — append at the latest end.
        latest_end = max(st + self.tubes[tid].duration for tid, st in current.items())
        return max(0.0, min(latest_end, max_start))

    def _create_input_vector(self) -> torch.Tensor:
        features: List[float] = []
        all_ids = sorted(self.tubes.keys())
        max_dur = max(
            (self.tubes[tid].duration for tid in all_ids),
            default=1.0,
        )
        if max_dur < 1e-6:
            max_dur = 1.0

        for tid in all_ids:
            is_placed = 1.0 if tid in self.placed_tubes else 0.0
            start = self.placed_tubes.get(tid, 0.0)
            norm_start = start / self.total_seconds if self.total_seconds > 0 else 0.0
            norm_dur = self.tubes[tid].duration / max_dur
            features.extend([is_placed, norm_start, norm_dur])

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def compute_reward_metric(self) -> float:
        """Compute reward using corrected per-frame 3D collision.

        Weights match Energy / PSO so all three optimizers minimize the same
        objective. Default metric is centroid-distance hinge — pairs farther
        than ``radius`` pixels apart are free, so the optimizer can pack
        spatially-disjoint tubes in parallel.
        """
        if not self.placed_tubes:
            return 0.0
        return compute_energy(
            self.tubes,
            self.placed_tubes,
            w_duration=1.0,
            w_collision=10.0,
            w_activity=10.0,
            w_chronology=self.w_chronology,
            chronology_M=self.chronology_M,
            method=self.collision_method,
            sigma=self.sigma,
            radius=self.radius,
            video_length=float(self.total_seconds),
            sample_step=1,
        )

    def backpropagate(self, value: float) -> None:
        self.visits += 1
        self.q_value_sum += value
        if self.parent:
            self.parent.backpropagate(value)


def mcts_search(root: MCTSNode, num_simulations: int) -> MCTSNode:
    for _ in range(num_simulations):
        node = root
        while not node.is_terminal():
            if not node.untried_actions:
                child = node.best_child()
                if child is None:
                    break
                node = child
            else:
                break

        value = 0.0
        if not node.is_terminal():
            if node.untried_actions:
                expanded = node.expand()
                if expanded:
                    node = expanded
                    value = node.nn_value
                else:
                    value = node.nn_value
            else:
                value = node.nn_value
        else:
            value = -node.compute_reward_metric()
        node.backpropagate(value)
    return root


def _self_play_episode(
    model: TubeNet,
    tubes: Dict[int, Tube],
    total_seconds: float,
    device: str,
    collision_method: str,
    sigma: float,
    radius: float,
    w_chronology: float,
    chronology_M: float,
    mcts_sims: int,
    video_size: Tuple[int, int],
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    slot_tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root = MCTSNode(
        placed_tubes={},
        remaining_tubes=list(tubes.keys()),
        tubes=tubes,
        total_seconds=total_seconds,
        model=model,
        device=device,
        collision_method=collision_method,
        sigma=sigma,
        radius=radius,
        w_chronology=w_chronology,
        chronology_M=chronology_M,
        size=video_size,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        slot_tolerance=slot_tolerance,
    )
    mcts_search(root, num_simulations=mcts_sims)

    num_actions = len(tubes)
    policy_target = torch.zeros(num_actions, dtype=torch.float32)

    if root.children:
        total_visits = sum(c.visits for c in root.children if c.visits > 0)
        if total_visits > 0:
            for child in root.children:
                if child.action_taken is not None and child.action_taken < num_actions:
                    policy_target[child.action_taken] = child.visits / total_visits

    value_target = torch.tensor([root.get_q_value()], dtype=torch.float32)
    return root._create_input_vector().cpu(), policy_target.cpu(), value_target.cpu()


def _train_model(
    model: TubeNet,
    optimizer: torch.optim.Optimizer,
    training_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 10,
    batch_size: int = 32,
    device: str = "cpu",
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
    for epoch in range(epochs):
        for batch_s, batch_p, batch_v in loader:
            batch_s, batch_p, batch_v = batch_s.to(device), batch_p.to(device), batch_v.to(device)
            optimizer.zero_grad()
            pred_p, pred_v = model(batch_s)
            loss = F.cross_entropy(pred_p, batch_p) + F.mse_loss(pred_v, batch_v)
            loss.backward()
            optimizer.step()
    model.eval()


class MCTSOptimizer(BaseOptimizer):
    """MCTS+AlphaZero optimizer with corrected per-frame collision detection."""

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

    def optimize(self, tubes: Dict[int, Tube], video_length_frames: int) -> Dict[int, float]:
        if not tubes:
            return {}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"MCTS optimizer using device: {device}")

        # Convert frames -> seconds. tube.duration is in seconds, so MCTS needs
        # to operate in seconds throughout (was mixing units before).
        total_seconds = float(video_length_frames) / max(self.fps, 1e-6)
        # Tighten the search horizon so MCTS doesn't try to schedule across the
        # full augmented timeline (which produces multi-thousand-second
        # synopses). target_duration ~= 3x max tube duration matches Energy.
        max_tube_dur = max((t.duration for t in tubes.values()), default=1.0)
        n = len(tubes)
        target_duration = max_tube_dur * min(3.0, max(1.5, n / 10.0))
        # Use the larger of target_duration and total_seconds as the hard
        # ceiling so the scheduler has slack if collisions force expansion.
        search_horizon = max(target_duration, total_seconds)
        log.info(
            f"MCTS horizon: target={target_duration:.1f}s, "
            f"hard={search_horizon:.1f}s (video={total_seconds:.1f}s)"
        )

        chronology_M = auto_tune_chronology_M(tubes) if self.w_chronology > 0 else 0.0
        if self.w_chronology > 0:
            log.info(f"Chronology weight: {self.w_chronology:.3f}, auto-tuned M: {chronology_M:.1f}s")

        num_tubes = len(tubes)
        nn_input = num_tubes * 3
        model = TubeNet(nn_input, num_actions=num_tubes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Self-play training
        all_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for episode in range(self.num_training_episodes):
            log.info(f"Self-play episode {episode+1}/{self.num_training_episodes}")
            model.eval()
            for _ in tqdm(range(self.games_per_episode), desc=f"Games Ep.{episode+1}"):
                data = _self_play_episode(
                    model, tubes, search_horizon, device,
                    self.collision_method, self.sigma, self.radius,
                    self.w_chronology, chronology_M,
                    self.mcts_sims_training, self.video_size,
                    self.c_puct, self.dirichlet_alpha, self.dirichlet_epsilon,
                    self.slot_tolerance,
                )
                all_data.append(data)
            _train_model(model, opt, all_data, self.training_epochs, self.batch_size, device)

        # Final optimization
        log.info("Running final MCTS optimization...")
        model.eval()
        placements: Dict[int, float] = {}
        remaining = list(tubes.keys())

        pbar = tqdm(range(num_tubes), desc="MCTS placing", unit="tube")
        for _ in pbar:
            if not remaining:
                break
            root = MCTSNode(
                placed_tubes=placements.copy(),
                remaining_tubes=remaining.copy(),
                tubes=tubes,
                total_seconds=search_horizon,
                model=model,
                device=device,
                collision_method=self.collision_method,
                sigma=self.sigma,
                radius=self.radius,
                w_chronology=self.w_chronology,
                chronology_M=chronology_M,
                size=self.video_size,
                c_puct=self.c_puct,
                dirichlet_alpha=0.0,
                dirichlet_epsilon=0.0,
                slot_tolerance=self.slot_tolerance,
            )
            mcts_search(root, num_simulations=self.mcts_sims_final)

            if not root.children:
                log.warning("No children from MCTS, stopping.")
                break

            best = max(root.children, key=lambda c: c.visits)
            action = best.action_taken
            placements[action] = best.placed_tubes[action]
            remaining.remove(action)
            pbar.set_postfix({"tid": action, "t": f"{placements[action]:.1f}s"})

        # Handle unplaced tubes
        if remaining:
            log.warning(f"{len(remaining)} tubes unplaced, using heuristic.")
            temp = MCTSNode(
                placed_tubes=placements.copy(),
                remaining_tubes=remaining.copy(),
                tubes=tubes,
                total_seconds=search_horizon,
                model=None,
                device="cpu",
                collision_method=self.collision_method,
                sigma=self.sigma,
                radius=self.radius,
                w_chronology=self.w_chronology,
                chronology_M=chronology_M,
                slot_tolerance=self.slot_tolerance,
            )
            for tid in remaining:
                start = temp._find_earliest_start_time(tid, placements)
                placements[tid] = start

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

"""Particle Swarm Optimization for tube placement.

GPU-accelerated: the swarm position matrix is a torch tensor on the chosen
device, and fitness is evaluated for all particles in a single batched
``compute_energy_torch`` call.
"""

import logging
import os
from typing import Dict

import torch

from video_synopsis.data.types import Tube
from video_synopsis.optimization.base import BaseOptimizer
from video_synopsis.optimization.collision import (
    TubeBatch,
    compute_energy_torch,
    pick_device,
)
from video_synopsis.optimization.visualize import save_initial_vs_optimized

log = logging.getLogger(__name__)


class PSOOptimizer(BaseOptimizer):
    """Optimizes tube start times using Particle Swarm Optimization.

    Population-based metaheuristic; explores the search space broadly
    without gradient computation or neural network overhead.
    """

    def __init__(
        self,
        num_particles: int = 30,
        max_iterations: int = 500,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        collision_method: str = "iou",
        sigma: float = 50.0,
        w_duration: float = 1.0,
        w_collision: float = 1000.0,
        w_activity: float = 10.0,
        sample_step: int = 1,
        output_dir: str = "optimized_tubes_pso",
        device: str = "",
        chunk_size: int = 16,
        fps: float = 30.0,
    ):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.collision_method = collision_method
        self.sigma = sigma
        self.w_duration = w_duration
        self.w_collision = w_collision
        self.w_activity = w_activity
        self.sample_step = sample_step
        self.output_dir = output_dir
        self.device_pref = device
        self.chunk_size = chunk_size
        self.fps = float(fps)

    def optimize(self, tubes: Dict[int, Tube], video_length_frames: int) -> Dict[int, float]:
        if not tubes:
            return {}

        device = pick_device(self.device_pref or None)
        log.info(f"PSOOptimizer using device: {device}")

        tube_ids = sorted(tubes.keys())
        n_tubes = len(tube_ids)
        video_length = float(video_length_frames) / max(self.fps, 1e-6)
        P = self.num_particles

        batch = TubeBatch(tubes, device=device)
        sample_count = max(8, 128 // max(self.sample_step, 1))

        # Init range stays tight (target_duration); hard bound is video_length.
        # Activity penalty at compute_energy_torch is the soft compression
        # budget — collision (high w_collision + IoU) keeps tubes apart.
        max_tube_dur = float(batch.durations.max().item()) if n_tubes > 0 else 0.0
        target_duration = max_tube_dur * min(3.0, max(1.5, n_tubes / 10.0))
        log.info(
            f"Init range: [0, target_duration={target_duration:.1f}s]; "
            f"hard bound: {video_length:.1f}s"
        )

        upper = (video_length - batch.durations).clamp(min=0.0)   # [N]
        lower = torch.zeros_like(upper)
        init_upper = (target_duration - batch.durations).clamp(min=0.0)

        gen = torch.Generator(device=device).manual_seed(0) if device.type != "mps" else None

        def _rand(*shape):
            # MPS doesn't support generator-based rand on all torch versions.
            if gen is None:
                return torch.rand(shape, device=device, dtype=batch.dtype)
            return torch.rand(shape, device=device, dtype=batch.dtype, generator=gen)

        positions = _rand(P, n_tubes) * init_upper.unsqueeze(0) + lower.unsqueeze(0)
        velocities = torch.zeros_like(positions)

        personal_best_pos = positions.clone()
        personal_best_fit = torch.full((P,), float("inf"), device=device, dtype=batch.dtype)

        global_best_pos = positions[0].clone()
        global_best_fit = torch.tensor(float("inf"), device=device, dtype=batch.dtype)

        for iteration in range(self.max_iterations):
            w_current = self.w - (self.w - 0.4) * iteration / max(1, self.max_iterations - 1)

            fitness = compute_energy_torch(
                batch, positions,
                w_duration=self.w_duration,
                w_collision=self.w_collision,
                w_activity=self.w_activity,
                method=self.collision_method,
                sigma=self.sigma,
                video_length=target_duration,
                sample_count=sample_count,
                chunk_size=self.chunk_size,
            )  # [P]

            improved = fitness < personal_best_fit
            if improved.any():
                personal_best_fit = torch.where(improved, fitness, personal_best_fit)
                personal_best_pos = torch.where(improved.unsqueeze(-1), positions, personal_best_pos)

            best_idx = int(torch.argmin(fitness).item())
            if fitness[best_idx] < global_best_fit:
                global_best_fit = fitness[best_idx].clone()
                global_best_pos = positions[best_idx].clone()

            r1 = _rand(P, n_tubes)
            r2 = _rand(P, n_tubes)
            velocities = (
                w_current * velocities
                + self.c1 * r1 * (personal_best_pos - positions)
                + self.c2 * r2 * (global_best_pos.unsqueeze(0) - positions)
            )
            positions = positions + velocities
            positions = torch.minimum(torch.maximum(positions, lower.unsqueeze(0)), upper.unsqueeze(0))

            if (iteration + 1) % 50 == 0:
                log.info(
                    f"PSO iteration {iteration + 1}/{self.max_iterations}, "
                    f"best energy: {float(global_best_fit.item()):.4f}"
                )

        best_np = global_best_pos.detach().cpu().numpy()
        result = {tid: float(best_np[i]) for i, tid in enumerate(tube_ids)}
        log.info(f"PSO optimization complete. Best energy: {float(global_best_fit.item()):.4f}")

        path = os.path.join(self.output_dir, "pso_optimized_plot.png")
        save_initial_vs_optimized(tubes, result, path, method_name="PSO")

        return result

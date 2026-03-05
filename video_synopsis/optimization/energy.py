"""Gradient-based energy optimizer using per-frame 3D collision detection."""

import logging
from typing import Dict

import numpy as np
import torch

from video_synopsis.data.types import Tube
from video_synopsis.optimization.base import BaseOptimizer
from video_synopsis.optimization.collision import compute_energy

log = logging.getLogger(__name__)


class EnergyOptimizer(BaseOptimizer):
    """Optimizes tube start times by minimizing an energy function via gradient descent.

    Uses the corrected per-frame 3D collision detection instead of static union boxes.
    """

    def __init__(
        self,
        epochs: int = 2000,
        lr: float = 0.5,
        collision_method: str = "repulsion",
        sigma: float = 50.0,
        w_duration: float = 1.0,
        w_collision: float = 10.0,
        w_activity: float = 0.1,
        sample_step: int = 2,
    ):
        self.epochs = epochs
        self.lr = lr
        self.collision_method = collision_method
        self.sigma = sigma
        self.w_duration = w_duration
        self.w_collision = w_collision
        self.w_activity = w_activity
        self.sample_step = sample_step

    def optimize(self, tubes: Dict[int, Tube], video_length_frames: int) -> Dict[int, float]:
        """Optimize tube placements via differentiable energy minimization.

        Args:
            tubes: Dict of tube_id -> Tube.
            video_length_frames: Total frames in the original video.

        Returns:
            Dict of tube_id -> optimized start time.
        """
        if not tubes:
            return {}

        tube_ids = sorted(tubes.keys())
        n = len(tube_ids)

        # Initialize start times from original timestamps
        init_starts = []
        for tid in tube_ids:
            tube = tubes[tid]
            init_starts.append(tube.start_time)

        start_params = torch.tensor(init_starts, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([start_params], lr=self.lr)

        video_length = float(video_length_frames)
        best_energy = float("inf")
        best_starts = start_params.detach().clone()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Build starts dict from current params
            starts = {}
            for i, tid in enumerate(tube_ids):
                starts[tid] = start_params[i].item()

            # Compute energy (not differentiable through collision sampling,
            # so we use finite differences for gradient estimation)
            energy_val = compute_energy(
                tubes, starts,
                w_duration=self.w_duration,
                w_collision=self.w_collision,
                w_activity=self.w_activity,
                method=self.collision_method,
                sigma=self.sigma,
                video_length=video_length,
                sample_step=self.sample_step,
            )

            if energy_val < best_energy:
                best_energy = energy_val
                best_starts = start_params.detach().clone()

            # Numerical gradient via finite differences
            eps = 0.5
            grad = torch.zeros_like(start_params)
            for i in range(n):
                starts_plus = dict(starts)
                starts_plus[tube_ids[i]] = starts[tube_ids[i]] + eps
                e_plus = compute_energy(
                    tubes, starts_plus,
                    w_duration=self.w_duration,
                    w_collision=self.w_collision,
                    w_activity=self.w_activity,
                    method=self.collision_method,
                    sigma=self.sigma,
                    video_length=video_length,
                    sample_step=self.sample_step,
                )
                grad[i] = (e_plus - energy_val) / eps

            # Manual gradient step
            with torch.no_grad():
                start_params -= self.lr * grad
                # Clamp to valid range
                for i in range(n):
                    dur = tubes[tube_ids[i]].duration
                    start_params[i] = start_params[i].clamp(0.0, max(0.0, video_length - dur))

            if (epoch + 1) % 200 == 0:
                log.info(f"Epoch {epoch+1}/{self.epochs}, Energy: {energy_val:.4f}, Best: {best_energy:.4f}")

        # Build final result from best
        result = {}
        for i, tid in enumerate(tube_ids):
            result[tid] = float(best_starts[i].item())

        log.info(f"Energy optimization complete. Best energy: {best_energy:.4f}")
        return result

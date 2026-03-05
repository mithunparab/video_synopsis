"""Gradient-based energy optimizer using per-frame 3D collision detection."""

import logging
from typing import Dict

import numpy as np

from video_synopsis.data.types import Tube
from video_synopsis.optimization.base import BaseOptimizer
from video_synopsis.optimization.collision import compute_energy

log = logging.getLogger(__name__)


class EnergyOptimizer(BaseOptimizer):
    """Optimizes tube start times by minimizing an energy function.

    The key insight for video synopsis: start with all tubes packed at t=0,
    then nudge them apart just enough to reduce spatial collisions.
    This produces a short, dense synopsis where multiple tubes play
    simultaneously (non-chronological).
    """

    def __init__(
        self,
        epochs: int = 2000,
        lr: float = 0.1,
        collision_method: str = "repulsion",
        sigma: float = 50.0,
        w_duration: float = 10.0,
        w_collision: float = 1.0,
        w_activity: float = 0.5,
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
        if not tubes:
            return {}

        tube_ids = sorted(tubes.keys())
        n = len(tube_ids)

        # Compute max tube duration for target synopsis length
        max_tube_dur = max(t.duration for t in tubes.values())
        # Target synopsis: a few multiples of the longest tube
        target_duration = max_tube_dur * min(3.0, max(1.5, n / 10.0))
        log.info(
            f"Optimizing {n} tubes. Max tube duration: {max_tube_dur:.1f}s, "
            f"Target synopsis: ~{target_duration:.1f}s"
        )

        # Initialize: stagger tubes slightly so they don't all start at 0
        # but pack them tightly within the target duration
        starts = np.zeros(n, dtype=np.float64)
        for i, tid in enumerate(tube_ids):
            starts[i] = (i / max(n - 1, 1)) * max(0, target_duration - tubes[tid].duration)

        best_energy = float("inf")
        best_starts = starts.copy()
        momentum = np.zeros_like(starts)
        mu = 0.9  # momentum coefficient

        for epoch in range(self.epochs):
            # Build starts dict
            starts_dict = {tid: starts[i] for i, tid in enumerate(tube_ids)}

            # Compute energy
            energy_val = compute_energy(
                tubes, starts_dict,
                w_duration=self.w_duration,
                w_collision=self.w_collision,
                w_activity=self.w_activity,
                method=self.collision_method,
                sigma=self.sigma,
                video_length=target_duration,
                sample_step=self.sample_step,
            )

            if energy_val < best_energy:
                best_energy = energy_val
                best_starts = starts.copy()

            # Numerical gradient via central finite differences (more stable)
            eps = 0.25
            grad = np.zeros(n, dtype=np.float64)
            for i in range(n):
                starts_plus = dict(starts_dict)
                starts_minus = dict(starts_dict)
                starts_plus[tube_ids[i]] = starts_dict[tube_ids[i]] + eps
                starts_minus[tube_ids[i]] = starts_dict[tube_ids[i]] - eps
                e_plus = compute_energy(
                    tubes, starts_plus,
                    w_duration=self.w_duration,
                    w_collision=self.w_collision,
                    w_activity=self.w_activity,
                    method=self.collision_method,
                    sigma=self.sigma,
                    video_length=target_duration,
                    sample_step=self.sample_step,
                )
                e_minus = compute_energy(
                    tubes, starts_minus,
                    w_duration=self.w_duration,
                    w_collision=self.w_collision,
                    w_activity=self.w_activity,
                    method=self.collision_method,
                    sigma=self.sigma,
                    video_length=target_duration,
                    sample_step=self.sample_step,
                )
                grad[i] = (e_plus - e_minus) / (2 * eps)

            # Gradient descent with momentum
            momentum = mu * momentum + self.lr * grad
            starts -= momentum

            # Clamp: keep starts within [0, target_duration]
            for i in range(n):
                dur = tubes[tube_ids[i]].duration
                starts[i] = np.clip(starts[i], 0.0, max(0.0, target_duration - dur))

            # Decay learning rate
            if (epoch + 1) % 500 == 0:
                self.lr *= 0.8

            if (epoch + 1) % 200 == 0:
                synopsis_len = max(
                    starts[i] + tubes[tube_ids[i]].duration for i in range(n)
                )
                log.info(
                    f"Epoch {epoch+1}/{self.epochs}, Energy: {energy_val:.4f}, "
                    f"Best: {best_energy:.4f}, Synopsis: {synopsis_len:.1f}s"
                )

        # Build final result from best
        result = {tid: float(best_starts[i]) for i, tid in enumerate(tube_ids)}

        synopsis_len = max(
            best_starts[i] + tubes[tube_ids[i]].duration for i in range(n)
        )
        log.info(
            f"Energy optimization complete. Best energy: {best_energy:.4f}, "
            f"Synopsis duration: {synopsis_len:.1f}s"
        )
        return result

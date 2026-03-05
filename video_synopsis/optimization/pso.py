"""Particle Swarm Optimization for tube placement."""

import logging
from typing import Dict

import numpy as np

from video_synopsis.data.types import Tube
from video_synopsis.optimization.base import BaseOptimizer
from video_synopsis.optimization.collision import compute_energy

log = logging.getLogger(__name__)


class PSOOptimizer(BaseOptimizer):
    """Optimizes tube start times using Particle Swarm Optimization.

    Population-based metaheuristic that explores the search space broadly
    without gradient computation or neural network overhead.
    """

    def __init__(
        self,
        num_particles: int = 30,
        max_iterations: int = 500,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        collision_method: str = "repulsion",
        sigma: float = 50.0,
        w_duration: float = 1.0,
        w_collision: float = 10.0,
        w_activity: float = 0.1,
        sample_step: int = 2,
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

    def optimize(self, tubes: Dict[int, Tube], video_length_frames: int) -> Dict[int, float]:
        if not tubes:
            return {}

        tube_ids = sorted(tubes.keys())
        n_tubes = len(tube_ids)
        video_length = float(video_length_frames)

        # Upper bounds per dimension: video_length - tube_duration
        upper_bounds = np.array([
            max(0.0, video_length - tubes[tid].duration) for tid in tube_ids
        ])
        lower_bounds = np.zeros(n_tubes)

        # Initialize particles randomly within valid ranges
        rng = np.random.default_rng()
        positions = rng.uniform(lower_bounds, upper_bounds + 1e-6, size=(self.num_particles, n_tubes))
        velocities = np.zeros((self.num_particles, n_tubes))

        # Personal bests
        personal_best_pos = positions.copy()
        personal_best_fit = np.full(self.num_particles, np.inf)

        # Global best
        global_best_pos = None
        global_best_fit = np.inf

        def _evaluate(position: np.ndarray) -> float:
            starts = {tid: position[i] for i, tid in enumerate(tube_ids)}
            return compute_energy(
                tubes, starts,
                w_duration=self.w_duration,
                w_collision=self.w_collision,
                w_activity=self.w_activity,
                method=self.collision_method,
                sigma=self.sigma,
                video_length=video_length,
                sample_step=self.sample_step,
            )

        for iteration in range(self.max_iterations):
            w_current = self.w - (self.w - 0.4) * iteration / max(1, self.max_iterations - 1)

            # Evaluate all particles
            for p in range(self.num_particles):
                fitness = _evaluate(positions[p])
                if fitness < personal_best_fit[p]:
                    personal_best_fit[p] = fitness
                    personal_best_pos[p] = positions[p].copy()
                if fitness < global_best_fit:
                    global_best_fit = fitness
                    global_best_pos = positions[p].copy()

            # Update velocities and positions
            r1 = rng.random((self.num_particles, n_tubes))
            r2 = rng.random((self.num_particles, n_tubes))
            velocities = (
                w_current * velocities
                + self.c1 * r1 * (personal_best_pos - positions)
                + self.c2 * r2 * (global_best_pos - positions)
            )
            positions += velocities

            # Clamp to valid range
            positions = np.clip(positions, lower_bounds, upper_bounds)

            if (iteration + 1) % 50 == 0:
                log.info(
                    f"PSO iteration {iteration + 1}/{self.max_iterations}, "
                    f"best energy: {global_best_fit:.4f}"
                )

        result = {tid: float(global_best_pos[i]) for i, tid in enumerate(tube_ids)}
        log.info(f"PSO optimization complete. Best energy: {global_best_fit:.4f}")
        return result

"""Gradient-based energy optimizer using per-frame 3D collision detection.

GPU-accelerated: all 2N+1 finite-difference perturbations per epoch are
evaluated in a single batched ``compute_energy_torch`` call, and the
position/momentum state lives on the chosen device.
"""

import logging
import os
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from video_synopsis.data.types import Tube
from video_synopsis.optimization.base import BaseOptimizer
from video_synopsis.optimization.collision import (
    TubeBatch,
    compute_energy_torch,
    pick_device,
)
from video_synopsis.optimization.visualize import save_initial_vs_optimized

log = logging.getLogger(__name__)


class EnergyOptimizer(BaseOptimizer):
    """Optimizes tube start times by minimizing an energy function.

    Starts with all tubes packed near t=0, then nudges them apart just
    enough to reduce spatial collisions. Produces a short, dense synopsis
    where multiple tubes play simultaneously (non-chronological).
    """

    def __init__(
        self,
        epochs: int = 2000,
        lr: float = 0.1,
        collision_method: str = "iou",
        sigma: float = 50.0,
        w_duration: float = 1.0,
        w_collision: float = 1000.0,
        w_activity: float = 10.0,
        sample_step: int = 1,
        output_dir: str = "optimized_tubes_energy",
        device: str = "",
        chunk_size: int = 16,
        fps: float = 30.0,
    ):
        self.epochs = epochs
        self.lr = lr
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
        log.info(f"EnergyOptimizer using device: {device}")

        tube_ids = sorted(tubes.keys())
        n = len(tube_ids)

        max_tube_dur = max(t.duration for t in tubes.values())
        target_duration = max_tube_dur * min(3.0, max(1.5, n / 10.0))
        log.info(
            f"Optimizing {n} tubes. Max tube duration: {max_tube_dur:.1f}s, "
            f"Target synopsis: ~{target_duration:.1f}s"
        )

        batch = TubeBatch(tubes, device=device)
        sample_count = max(8, 128 // max(self.sample_step, 1))

        # Initial placement: stagger tubes within target_duration so the
        # finite-diff gradient sees overlap from the start.
        durations_np = batch.durations.detach().cpu().numpy()
        starts0 = np.zeros(n, dtype=np.float32)
        denom = max(n - 1, 1)
        for i in range(n):
            starts0[i] = (i / denom) * max(0.0, target_duration - durations_np[i])

        starts = torch.from_numpy(starts0).to(device)
        # Hard bounds: [0, video_length - duration]. Activity penalty (against
        # target_duration) creates the soft compression budget; collision keeps
        # tubes apart. Tubes can drift past target_duration if collision forces
        # it. video_length comes in frames; convert to seconds so it lives in
        # the same unit as tube.duration.
        video_length_seconds = float(video_length_frames) / max(self.fps, 1e-6)
        upper = (video_length_seconds - batch.durations).clamp(min=0.0)
        zero = torch.zeros((), device=device, dtype=batch.dtype)

        momentum = torch.zeros_like(starts)
        mu = 0.9
        eps = 0.25
        lr = self.lr

        best_energy = float("inf")
        best_starts = starts.clone()

        # Precompute the batched perturbation matrix: [2N+1, N]
        # row 0 = baseline, rows 1..N = +eps on dim k, rows N+1..2N = -eps on dim k.
        eye = torch.eye(n, device=device, dtype=batch.dtype) * eps

        pbar = tqdm(range(self.epochs), desc="Energy", unit="ep")
        for epoch in pbar:
            perturb = torch.cat(
                [starts.unsqueeze(0), starts.unsqueeze(0) + eye, starts.unsqueeze(0) - eye],
                dim=0,
            )  # [2N+1, N]

            energies = compute_energy_torch(
                batch, perturb,
                w_duration=self.w_duration,
                w_collision=self.w_collision,
                w_activity=self.w_activity,
                method=self.collision_method,
                sigma=self.sigma,
                video_length=float(target_duration),
                sample_count=sample_count,
                chunk_size=self.chunk_size,
            )  # [2N+1]

            energy_val = float(energies[0].item())
            if energy_val < best_energy:
                best_energy = energy_val
                best_starts = starts.clone()

            grad = (energies[1:n + 1] - energies[n + 1:2 * n + 1]) / (2 * eps)  # [N]

            momentum = mu * momentum + lr * grad
            starts = starts - momentum

            # Clamp to [0, target_duration - tube_duration] per tube.
            starts = torch.minimum(torch.maximum(starts, zero), upper)

            if (epoch + 1) % 500 == 0:
                lr *= 0.8

            if (epoch + 1) % 50 == 0:
                synopsis_len = float((starts + batch.durations).max().item())
                pbar.set_postfix({
                    "E": f"{energy_val:.1f}",
                    "best": f"{best_energy:.1f}",
                    "syn": f"{synopsis_len:.1f}s",
                })

        best_np = best_starts.detach().cpu().numpy()
        result = {tid: float(best_np[i]) for i, tid in enumerate(tube_ids)}

        synopsis_len = float(
            (best_starts + batch.durations).max().item()
        )
        log.info(
            f"Energy optimization complete. Best energy: {best_energy:.4f}, "
            f"Synopsis duration: {synopsis_len:.1f}s"
        )

        path = os.path.join(self.output_dir, "energy_optimized_plot.png")
        save_initial_vs_optimized(tubes, result, path, method_name="Energy (gradient)")

        return result

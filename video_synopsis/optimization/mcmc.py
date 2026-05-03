"""Parallel Metropolis-Hastings MCMC optimizer for tube start times.

Inspired by Nie et al. (TIP 2019, "Collision-Free Video Synopsis Incorporating
Object Speed and Size Changes"), but adapted for our framework:

  * Their version is single-chain, 30M serial steps in C++. We run B parallel
    chains on GPU, each making one proposal per step, fitness evaluated for all
    chains in a single batched ``compute_energy_torch`` call.
  * They optimise (start, speed[paragraph], size[paragraph]). This first cut
    only optimises ``start`` — adding paragraph speed/size requires extending
    the collision kernel and the renderer, which is a larger surface area.
  * Their collision metric is bbox-area overlap; we keep our centroid-distance
    hinge from the rest of the framework.

The optimiser shares the same energy function as Energy/PSO, so all four
methods optimise the same objective and are directly comparable.
"""

import logging
import math
import os
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from video_synopsis.data.types import Tube
from video_synopsis.optimization.base import BaseOptimizer
from video_synopsis.optimization.collision import (
    TubeBatch,
    auto_tune_chronology_M,
    compute_energy_torch,
    pick_device,
)
from video_synopsis.optimization.visualize import save_initial_vs_optimized

log = logging.getLogger(__name__)


class MCMCOptimizer(BaseOptimizer):
    """Ensemble Metropolis-Hastings sampler over tube start times.

    At each step every chain independently picks one tube and proposes a
    Gaussian perturbation. Acceptance follows ``min(1, exp(-γ ΔE))`` with γ
    annealed as ``γ = anneal_factor / E_best``.
    """

    def __init__(
        self,
        num_chains: int = 32,
        num_steps: int = 50_000,
        proposal_std: float = 5.0,
        global_jump_prob: float = 0.1,
        anneal_factor: float = 600.0,
        collision_method: str = "centroid",
        sigma: float = 50.0,
        radius: float = 30.0,
        w_duration: float = 1.0,
        w_collision: float = 10.0,
        w_activity: float = 10.0,
        w_chronology: float = 0.0,
        sample_step: int = 1,
        output_dir: str = "optimized_tubes_mcmc",
        device: str = "",
        chunk_size: int = 16,
        fps: float = 30.0,
        seed: int = 0,
    ):
        self.num_chains = int(num_chains)
        self.num_steps = int(num_steps)
        self.proposal_std = float(proposal_std)
        self.global_jump_prob = float(global_jump_prob)
        self.anneal_factor = float(anneal_factor)
        self.collision_method = collision_method
        self.sigma = sigma
        self.radius = radius
        self.w_duration = w_duration
        self.w_collision = w_collision
        self.w_activity = w_activity
        self.w_chronology = w_chronology
        self.sample_step = sample_step
        self.output_dir = output_dir
        self.device_pref = device
        self.chunk_size = chunk_size
        self.fps = float(fps)
        self.seed = int(seed)

    def optimize(self, tubes: Dict[int, Tube], video_length_frames: int) -> Dict[int, float]:
        if not tubes:
            return {}

        device = pick_device(self.device_pref or None)
        log.info(f"MCMCOptimizer using device: {device}")

        tube_ids = sorted(tubes.keys())
        N = len(tube_ids)
        video_length = float(video_length_frames) / max(self.fps, 1e-6)

        batch = TubeBatch(tubes, device=device)
        sample_count = max(8, 128 // max(self.sample_step, 1))
        chronology_M = auto_tune_chronology_M(tubes) if self.w_chronology > 0 else 0.0
        if self.w_chronology > 0:
            log.info(f"Chronology weight: {self.w_chronology:.3f}, M: {chronology_M:.1f}s")

        max_tube_dur = float(batch.durations.max().item()) if N > 0 else 0.0
        target_duration = max_tube_dur * min(3.0, max(1.5, N / 10.0))
        log.info(
            f"Init range: [0, target_duration={target_duration:.1f}s]; "
            f"hard bound: {video_length:.1f}s"
        )

        upper = (video_length - batch.durations).clamp(min=0.0)               # [N]
        init_upper = (target_duration - batch.durations).clamp(min=0.0)
        lower = torch.zeros_like(upper)

        # Cross-platform RNG. CUDA generator path is fastest; MPS lacks generator
        # support on all torch builds, so fall back to ungenerated rand.
        gen = torch.Generator(device=device).manual_seed(self.seed) if device.type != "mps" else None

        def _rand(*shape):
            if gen is None:
                return torch.rand(shape, device=device, dtype=batch.dtype)
            return torch.rand(shape, device=device, dtype=batch.dtype, generator=gen)

        def _randn(*shape):
            if gen is None:
                return torch.randn(shape, device=device, dtype=batch.dtype)
            return torch.randn(shape, device=device, dtype=batch.dtype, generator=gen)

        def _randint(low, high, shape):
            if gen is None:
                return torch.randint(low, high, shape, device=device)
            return torch.randint(low, high, shape, device=device, generator=gen)

        # B parallel chains, each a tentative placement
        B = self.num_chains
        states = _rand(B, N) * init_upper.unsqueeze(0) + lower.unsqueeze(0)    # [B, N]

        def _energy(positions: torch.Tensor) -> torch.Tensor:
            return compute_energy_torch(
                batch, positions,
                w_duration=self.w_duration,
                w_collision=self.w_collision,
                w_activity=self.w_activity,
                w_chronology=self.w_chronology,
                chronology_M=chronology_M,
                method=self.collision_method,
                sigma=self.sigma,
                radius=self.radius,
                video_length=target_duration,
                sample_count=sample_count,
                chunk_size=self.chunk_size,
            )

        energies = _energy(states)                                              # [B]

        best_idx = int(torch.argmin(energies).item())
        best_state = states[best_idx].clone()
        best_energy = float(energies[best_idx].item())
        # γ scaling: keep acceptance probability sane initially. Following the
        # paper's heuristic.
        gamma = self.anneal_factor / max(best_energy, 1e-3)

        chain_idx = torch.arange(B, device=device)
        accept_count = 0
        pbar = tqdm(range(self.num_steps), desc="MCMC", unit="step")
        for step in pbar:
            # Each chain picks one tube to perturb; mostly local (Gaussian),
            # occasionally a global jump to a uniform random valid position.
            tube_idx = _randint(0, N, (B,))                                    # [B]
            local_delta = _randn(B) * self.proposal_std                         # [B]
            global_pos = _rand(B) * upper.gather(0, tube_idx)                   # [B]
            jump_mask = (_rand(B) < self.global_jump_prob)
            current = states[chain_idx, tube_idx]
            new_val = torch.where(jump_mask, global_pos, current + local_delta)
            # Clamp to per-tube hard bounds
            new_val = torch.minimum(
                torch.maximum(new_val, torch.zeros_like(new_val)),
                upper.gather(0, tube_idx),
            )

            proposal = states.clone()
            proposal[chain_idx, tube_idx] = new_val

            new_energies = _energy(proposal)
            delta_e = new_energies - energies                                   # [B]

            # Metropolis acceptance. Bound the exponent before exp to avoid
            # overflow when ΔE is large and negative.
            log_alpha = (-gamma * delta_e).clamp(max=0.0)
            accept = (_rand(B).log() < log_alpha)
            accept_count += int(accept.sum().item())

            states = torch.where(accept.unsqueeze(-1), proposal, states)
            energies = torch.where(accept, new_energies, energies)

            cur_best_idx = int(torch.argmin(energies).item())
            cur_best_e = float(energies[cur_best_idx].item())
            if cur_best_e < best_energy:
                best_energy = cur_best_e
                best_state = states[cur_best_idx].clone()
                # Re-anneal: keep acceptance probability calibrated as the
                # energy scale shrinks.
                gamma = self.anneal_factor / max(best_energy, 1e-3)

            if (step + 1) % max(1, self.num_steps // 200) == 0:
                acc_rate = accept_count / max(1, (step + 1) * B)
                synopsis_len = float((best_state + batch.durations).max().item())
                pbar.set_postfix({
                    "best": f"{best_energy:.1f}",
                    "syn": f"{synopsis_len:.1f}s",
                    "acc": f"{acc_rate:.2%}",
                    "γ": f"{gamma:.2g}",
                })

        best_np = best_state.detach().cpu().numpy()
        result = {tid: float(best_np[i]) for i, tid in enumerate(tube_ids)}
        synopsis_len = float((best_state + batch.durations).max().item())
        log.info(
            f"MCMC complete. Best energy: {best_energy:.4f}, "
            f"synopsis: {synopsis_len:.1f}s, accept rate: "
            f"{accept_count / max(1, self.num_steps * B):.2%}"
        )

        path = os.path.join(self.output_dir, "mcmc_optimized_plot.png")
        save_initial_vs_optimized(tubes, result, path, method_name="MCMC")

        return result

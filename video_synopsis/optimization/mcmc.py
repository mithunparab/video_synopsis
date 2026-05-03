"""Parallel Metropolis-Hastings MCMC over (start, speed[paragraph], size[paragraph]).

Inspired by Nie et al. (TIP 2019). Differences from the paper:

  * Their version is single-chain, 30M serial steps in C++. We run B parallel
    chains on GPU; fitness for all chains is evaluated in a single batched
    ``compute_energy_torch`` call.
  * Their collision metric is bbox-area overlap. We use the centroid-distance
    hinge with size-aware effective radius — shrinking ``size[k]`` makes the
    effective collision radius smaller, so two close-centroid tubes can run
    concurrently when both shrink.
  * Speed and size are per paragraph (~``paragraph_seconds`` of source). With
    ``optimize_speed=False, optimize_size=False`` the optimiser reduces to
    start-time-only and matches the previous behaviour exactly (legacy mode).
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
    auto_tune_chronology_M,
    compute_energy_torch,
    pick_device,
)
from video_synopsis.optimization.schedule import Schedule
from video_synopsis.optimization.visualize import save_initial_vs_optimized

log = logging.getLogger(__name__)


class MCMCOptimizer(BaseOptimizer):
    """Ensemble Metropolis-Hastings sampler over tube placements.

    Each step every chain picks a tube and, with configurable mix, perturbs
    its synopsis start, one paragraph's speed, or one paragraph's size. The
    proposal is accepted with prob ``min(1, exp(-γ ΔE))``; γ is annealed as
    ``γ = anneal_factor / E_best``.

    The best chain's full state is converted to ``Schedule`` objects accessible
    via ``self.last_schedules`` for the renderer to pick up. The base
    ``optimize`` return value is still ``Dict[int, float]`` (start times) for
    backward compatibility with stitchers that only care about start time.
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
        # Speed / size optimisation
        optimize_speed: bool = False,
        optimize_size: bool = False,
        paragraph_seconds: float = 2.0,
        speed_min: float = 0.5,
        speed_max: float = 4.0,
        size_min: float = 0.5,
        speed_proposal_std: float = 0.15,   # log-space stddev
        size_proposal_std: float = 0.05,
        w_speed_reg: float = 1.0,
        w_size_reg: float = 1.0,
        speed_reg_alpha: float = 2.0,
        size_reg_alpha: float = 2.0,
        # Move-mix
        speed_move_prob: float = 0.2,
        size_move_prob: float = 0.1,
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

        self.optimize_speed = bool(optimize_speed)
        self.optimize_size = bool(optimize_size)
        self.paragraph_seconds = float(paragraph_seconds)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.size_min = float(size_min)
        self.speed_proposal_std = float(speed_proposal_std)
        self.size_proposal_std = float(size_proposal_std)
        self.w_speed_reg = float(w_speed_reg)
        self.w_size_reg = float(w_size_reg)
        self.speed_reg_alpha = float(speed_reg_alpha)
        self.size_reg_alpha = float(size_reg_alpha)
        self.speed_move_prob = float(speed_move_prob) if optimize_speed else 0.0
        self.size_move_prob = float(size_move_prob) if optimize_size else 0.0

        self.sample_step = sample_step
        self.output_dir = output_dir
        self.device_pref = device
        self.chunk_size = chunk_size
        self.fps = float(fps)
        self.seed = int(seed)

        # Filled by ``optimize``. ``last_schedules`` carries the full per-tube
        # decision (start, speeds, sizes) for the renderer.
        self.last_schedules: Dict[int, Schedule] = {}

    def optimize(self, tubes: Dict[int, Tube], video_length_frames: int) -> Dict[int, float]:
        if not tubes:
            self.last_schedules = {}
            return {}

        device = pick_device(self.device_pref or None)
        log.info(f"MCMCOptimizer using device: {device}")
        log.info(
            f"optimise_speed={self.optimize_speed}, optimise_size={self.optimize_size}, "
            f"paragraph_seconds={self.paragraph_seconds}"
        )

        tube_ids = sorted(tubes.keys())
        N = len(tube_ids)
        video_length = float(video_length_frames) / max(self.fps, 1e-6)

        batch = TubeBatch(tubes, device=device, paragraph_seconds=self.paragraph_seconds)
        Kp = batch.K_max
        sample_count = max(8, 128 // max(self.sample_step, 1))
        chronology_M = auto_tune_chronology_M(tubes) if self.w_chronology > 0 else 0.0
        if self.w_chronology > 0:
            log.info(f"Chronology weight: {self.w_chronology:.3f}, M: {chronology_M:.1f}s")

        max_tube_dur = float(batch.durations.max().item()) if N > 0 else 0.0
        target_duration = max_tube_dur * min(3.0, max(1.5, N / 10.0))
        log.info(
            f"Init range: [0, target_duration={target_duration:.1f}s]; hard bound: {video_length:.1f}s; "
            f"K_max paragraphs: {Kp}"
        )

        # Hard upper bound on start: synopsis duration depends on speed, but use
        # native (speed=1) duration here for clamp; the activity term punishes
        # any drift past video_length anyway.
        upper_native = (video_length - batch.durations).clamp(min=0.0)         # [N]
        init_upper = (target_duration - batch.durations).clamp(min=0.0)
        zero = torch.zeros_like(upper_native)

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

        B = self.num_chains
        states = _rand(B, N) * init_upper.unsqueeze(0) + zero.unsqueeze(0)     # [B, N]
        speeds = torch.ones((B, N, Kp), device=device, dtype=batch.dtype)
        sizes = torch.ones((B, N, Kp), device=device, dtype=batch.dtype)

        # Mask for valid (tube, paragraph) cells — used to keep proposals from
        # touching padded paragraph slots beyond ``para_lengths[i]``.
        arange_kp = torch.arange(Kp, device=device).unsqueeze(0)               # [1, Kp]
        valid_NK = (arange_kp < batch.para_lengths.unsqueeze(-1))              # [N, Kp]

        def _energy(s_starts, s_speeds, s_sizes):
            # Pass speeds/sizes only when not all-ones, so the legacy fast path
            # kicks in for start-only mode.
            sp = s_speeds if self.optimize_speed else None
            sz = s_sizes if self.optimize_size else None
            return compute_energy_torch(
                batch, s_starts,
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
                speeds=sp, sizes=sz,
                w_speed_reg=self.w_speed_reg if self.optimize_speed else 0.0,
                w_size_reg=self.w_size_reg if self.optimize_size else 0.0,
                speed_reg_alpha=self.speed_reg_alpha,
                size_reg_alpha=self.size_reg_alpha,
            )

        energies = _energy(states, speeds, sizes)                              # [B]

        best_idx = int(torch.argmin(energies).item())
        best_state = states[best_idx].clone()
        best_speeds = speeds[best_idx].clone()
        best_sizes = sizes[best_idx].clone()
        best_energy = float(energies[best_idx].item())
        gamma = self.anneal_factor / max(best_energy, 1e-3)

        chain_idx = torch.arange(B, device=device)
        # Move-mix probabilities. Remainder is start-time move (the dominant move).
        p_speed = self.speed_move_prob
        p_size = self.size_move_prob
        p_start = max(0.0, 1.0 - p_speed - p_size)

        accept_count = 0
        pbar = tqdm(range(self.num_steps), desc="MCMC", unit="step")
        for step in pbar:
            tube_idx = _randint(0, N, (B,))                                    # [B]
            move_roll = _rand(B)
            do_speed = (move_roll < p_speed) if self.optimize_speed else torch.zeros(B, dtype=torch.bool, device=device)
            do_size  = (~do_speed) & ((move_roll < p_speed + p_size) if self.optimize_size else torch.zeros(B, dtype=torch.bool, device=device))
            do_start = ~(do_speed | do_size)

            # Per-chain: which paragraph to perturb (only used by speed/size moves)
            para_idx = _randint(0, max(Kp, 1), (B,))                           # [B]
            # Clamp para_idx to actual paragraph count of the chosen tube
            tube_pars = batch.para_lengths.gather(0, tube_idx)                 # [B]
            para_idx = torch.minimum(para_idx, (tube_pars - 1).clamp(min=0))

            # Build proposal as clones; only the perturbed cells differ
            prop_states = states.clone()
            prop_speeds = speeds.clone()
            prop_sizes = sizes.clone()

            # ---- start move (Gaussian + occasional global jump) ----
            local_delta = _randn(B) * self.proposal_std
            global_pos = _rand(B) * upper_native.gather(0, tube_idx)
            jump_mask = (_rand(B) < self.global_jump_prob)
            cur_start = states[chain_idx, tube_idx]
            new_start = torch.where(jump_mask, global_pos, cur_start + local_delta)
            new_start = torch.minimum(torch.maximum(new_start, torch.zeros_like(new_start)),
                                      upper_native.gather(0, tube_idx))

            # ---- speed move (log-space Gaussian) ----
            cur_speed = speeds[chain_idx, tube_idx, para_idx]
            log_delta = _randn(B) * self.speed_proposal_std
            new_speed = (cur_speed * torch.exp(log_delta)).clamp(min=self.speed_min, max=self.speed_max)

            # ---- size move (linear, clipped) ----
            cur_size = sizes[chain_idx, tube_idx, para_idx]
            size_delta = _randn(B) * self.size_proposal_std
            new_size = (cur_size + size_delta).clamp(min=self.size_min, max=1.0)

            # Apply only to chains that chose that move
            sel_start = do_start.nonzero(as_tuple=True)[0]
            if sel_start.numel():
                prop_states[sel_start, tube_idx[sel_start]] = new_start[sel_start]
            sel_speed = do_speed.nonzero(as_tuple=True)[0]
            if sel_speed.numel():
                prop_speeds[sel_speed, tube_idx[sel_speed], para_idx[sel_speed]] = new_speed[sel_speed]
            sel_size = do_size.nonzero(as_tuple=True)[0]
            if sel_size.numel():
                prop_sizes[sel_size, tube_idx[sel_size], para_idx[sel_size]] = new_size[sel_size]

            new_energies = _energy(prop_states, prop_speeds, prop_sizes)
            delta_e = new_energies - energies

            log_alpha = (-gamma * delta_e).clamp(max=0.0)
            accept = (_rand(B).log() < log_alpha)
            accept_count += int(accept.sum().item())

            mask3 = accept.view(B, 1, 1)
            states = torch.where(accept.unsqueeze(-1), prop_states, states)
            speeds = torch.where(mask3, prop_speeds, speeds)
            sizes = torch.where(mask3, prop_sizes, sizes)
            energies = torch.where(accept, new_energies, energies)

            cur_best_idx = int(torch.argmin(energies).item())
            cur_best_e = float(energies[cur_best_idx].item())
            if cur_best_e < best_energy:
                best_energy = cur_best_e
                best_state = states[cur_best_idx].clone()
                best_speeds = speeds[cur_best_idx].clone()
                best_sizes = sizes[cur_best_idx].clone()
                gamma = self.anneal_factor / max(best_energy, 1e-3)

            if (step + 1) % max(1, self.num_steps // 200) == 0:
                acc_rate = accept_count / max(1, (step + 1) * B)
                # Synopsis length under current best speeds:
                syn_dur_per_tube = (
                    (batch.src_para_durs / best_speeds.clamp(min=1e-6)) *
                    valid_NK.to(batch.dtype)
                ).sum(dim=-1)
                synopsis_len = float((best_state + syn_dur_per_tube).max().item())
                pbar.set_postfix({
                    "best": f"{best_energy:.1f}",
                    "syn": f"{synopsis_len:.1f}s",
                    "acc": f"{acc_rate:.2%}",
                })

        # Convert best chain to Schedule objects
        self.last_schedules = {}
        starts_np = best_state.detach().cpu().numpy()
        speeds_np = best_speeds.detach().cpu().numpy()
        sizes_np = best_sizes.detach().cpu().numpy()
        para_lengths_np = batch.para_lengths.detach().cpu().numpy()
        src_para_starts_np = batch.src_para_starts.detach().cpu().numpy()
        src_para_durs_np = batch.src_para_durs.detach().cpu().numpy()
        for i, tid in enumerate(tube_ids):
            K = int(para_lengths_np[i])
            self.last_schedules[tid] = Schedule(
                tube_id=int(tid),
                start=float(starts_np[i]),
                src_para_starts=src_para_starts_np[i, :K].copy().astype(np.float32),
                src_para_durs=src_para_durs_np[i, :K].copy().astype(np.float32),
                speeds=speeds_np[i, :K].copy().astype(np.float32),
                sizes=sizes_np[i, :K].copy().astype(np.float32),
            )

        result = {tid: float(starts_np[i]) for i, tid in enumerate(tube_ids)}
        syn_dur_per_tube = (
            (batch.src_para_durs / best_speeds.clamp(min=1e-6)) *
            valid_NK.to(batch.dtype)
        ).sum(dim=-1)
        synopsis_len = float((best_state + syn_dur_per_tube).max().item())
        log.info(
            f"MCMC complete. Best energy: {best_energy:.4f}, synopsis: {synopsis_len:.1f}s, "
            f"accept rate: {accept_count / max(1, self.num_steps * B):.2%}"
        )

        path = os.path.join(self.output_dir, "mcmc_optimized_plot.png")
        save_initial_vs_optimized(tubes, result, path, method_name="MCMC")

        return result

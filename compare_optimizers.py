"""Run every tube-placement optimizer on the same tube set and produce one
side-by-side comparison plot.

Inputs are pre-generated tubes saved as .npz files (see TubeArchive.save_all
or `--use_npz` in the main pipeline) — this avoids re-running detection and
tracking for every optimizer.

Usage (sequential):
    python compare_optimizers.py --tubes_npz_dir ./tubes_npz \\
        --output_dir ./comparison_out \\
        --methods energy,pso,mcts

Usage (parallel, one method per GPU):
    python compare_optimizers.py --tubes_npz_dir ./tubes_npz \\
        --output_dir ./comparison_out \\
        --methods energy,pso,mcts \\
        --parallel --gpus 0,1

Outputs:
    <output_dir>/<method>/<method>_optimized_plot.png  — initial vs optimized per method
    <output_dir>/<method>/placements.json              — written when --parallel is used
    <output_dir>/methods_comparison.png                — single figure with all methods
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from video_synopsis.data.tube_store import TubeArchive
from video_synopsis.data.types import Tube
from video_synopsis.optimization.visualize import save_methods_comparison


def _resolve_video_length(
    tubes_npz_dir: str,
    tubes: Dict[int, Tube],
    override: Optional[int],
) -> Tuple[int, str]:
    """Pick the video length used as the timeline upper bound.

    Priority:
      1. --video_length_frames override
      2. metadata.json::target_video_length_frames (written by tube_augment.py)
      3. fallback: largest frame_index across tubes (matches original videos)
    """
    if override and override > 0:
        return override, "override"

    meta_path = os.path.join(tubes_npz_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "target_video_length_frames" in meta:
            return int(meta["target_video_length_frames"]), f"metadata.json ({meta_path})"

    max_idx = 0
    for tube in tubes.values():
        if tube.num_frames == 0:
            continue
        max_idx = max(max_idx, int(tube.frame_indices_array.max()))
    return max_idx + 1, "max frame_index across tubes"


def _resolve_fps(tubes_npz_dir: str, override: Optional[float]) -> float:
    """fps for converting video_length_frames -> seconds inside optimizers."""
    if override and override > 0:
        return float(override)
    meta_path = os.path.join(tubes_npz_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if "fps" in meta:
            return float(meta["fps"])
    return 30.0


def _build_optimizer(name: str, output_dir: str, args, fps: float):
    name = name.lower()
    if name == "energy":
        from video_synopsis.optimization.energy import EnergyOptimizer
        return EnergyOptimizer(
            epochs=args.energy_epochs,
            collision_method=args.collision_method,
            sigma=args.sigma,
            radius=args.collision_radius,
            output_dir=output_dir,
            fps=fps,
        )
    if name == "pso":
        from video_synopsis.optimization.pso import PSOOptimizer
        return PSOOptimizer(
            num_particles=args.pso_num_particles,
            max_iterations=args.pso_max_iterations,
            collision_method=args.collision_method,
            sigma=args.sigma,
            radius=args.collision_radius,
            output_dir=output_dir,
            fps=fps,
        )
    if name == "mcts":
        from video_synopsis.optimization.mcts import MCTSOptimizer
        return MCTSOptimizer(
            num_training_episodes=args.mcts_training_episodes,
            games_per_episode=args.mcts_games_per_episode,
            mcts_sims_training=args.mcts_sims_training,
            mcts_sims_final=args.mcts_sims_final,
            collision_method=args.collision_method,
            sigma=args.sigma,
            radius=args.collision_radius,
            w_chronology=args.w_chronology,
            output_dir=output_dir,
            fps=fps,
        )
    if name == "mcmc":
        from video_synopsis.optimization.mcmc import MCMCOptimizer
        return MCMCOptimizer(
            num_chains=args.mcmc_chains,
            num_steps=args.mcmc_steps,
            proposal_std=args.mcmc_proposal_std,
            global_jump_prob=args.mcmc_global_jump_prob,
            optimize_speed=args.mcmc_optimize_speed,
            optimize_size=args.mcmc_optimize_size,
            paragraph_seconds=args.paragraph_seconds,
            speed_min=args.speed_min,
            speed_max=args.speed_max,
            size_min=args.size_min,
            w_speed_reg=args.w_speed_reg,
            w_size_reg=args.w_size_reg,
            collision_method=args.collision_method,
            sigma=args.sigma,
            radius=args.collision_radius,
            w_chronology=args.w_chronology,
            output_dir=output_dir,
            fps=fps,
        )
    raise ValueError(f"Unknown optimizer: {name!r} (expected energy|pso|mcts|mcmc)")


_PRETTY_NAME = {
    "energy": "Energy (gradient)",
    "pso": "PSO",
    "mcts": "MCTS+AlphaZero",
    "mcmc": "MCMC (parallel chains)",
}


def _placements_path(method_dir: str) -> str:
    return os.path.join(method_dir, "placements.json")


def _run_one_method(
    name: str,
    args,
    tubes: Dict[int, Tube],
    video_length: int,
    fps: float,
    log: logging.Logger,
) -> Tuple[Dict[int, float], float]:
    """Run a single optimizer in-process and return (placements, elapsed_seconds)."""
    pretty = _PRETTY_NAME.get(name, name)
    log.info(f"=== Running {pretty} ===")
    method_dir = os.path.join(args.output_dir, name)
    os.makedirs(method_dir, exist_ok=True)
    optimizer = _build_optimizer(name, method_dir, args, fps)
    t0 = time.time()
    placements = optimizer.optimize(tubes, video_length)
    elapsed = time.time() - t0
    log.info(f"{pretty} finished in {elapsed:.1f}s with {len(placements)} placements")
    return placements, elapsed


def _spawn_subprocess(
    method: str,
    gpu_id: Optional[int],
    args,
    log: logging.Logger,
) -> subprocess.Popen:
    """Re-invoke this script in --single_method mode pinned to one GPU."""
    method_dir = os.path.join(args.output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    log_path = os.path.join(method_dir, "stdout.log")

    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--tubes_npz_dir", args.tubes_npz_dir,
        "--output_dir", args.output_dir,
        "--methods", method,
        "--video_length_frames", str(args.video_length_frames),
        "--fps", str(args.fps),
        "--collision_method", args.collision_method,
        "--sigma", str(args.sigma),
        "--collision_radius", str(args.collision_radius),
        "--w_chronology", str(args.w_chronology),
        "--mcmc_chains", str(args.mcmc_chains),
        "--mcmc_steps", str(args.mcmc_steps),
        "--mcmc_proposal_std", str(args.mcmc_proposal_std),
        "--mcmc_global_jump_prob", str(args.mcmc_global_jump_prob),
        "--paragraph_seconds", str(args.paragraph_seconds),
        "--speed_min", str(args.speed_min),
        "--speed_max", str(args.speed_max),
        "--size_min", str(args.size_min),
        "--w_speed_reg", str(args.w_speed_reg),
        "--w_size_reg", str(args.w_size_reg),
        "--energy_epochs", str(args.energy_epochs),
        "--pso_num_particles", str(args.pso_num_particles),
        "--pso_max_iterations", str(args.pso_max_iterations),
        "--mcts_training_episodes", str(args.mcts_training_episodes),
        "--mcts_games_per_episode", str(args.mcts_games_per_episode),
        "--mcts_sims_training", str(args.mcts_sims_training),
        "--mcts_sims_final", str(args.mcts_sims_final),
        "--single_method",
    ]
    if getattr(args, "mcmc_optimize_speed", False):
        cmd.append("--mcmc_optimize_speed")
    if getattr(args, "mcmc_optimize_size", False):
        cmd.append("--mcmc_optimize_size")

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log.info(
        f"Spawning {method} on GPU {gpu_id if gpu_id is not None else 'cpu'}; "
        f"logs -> {log_path}"
    )
    log_fp = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env)


def _save_placements(method_dir: str, placements: Dict[int, float], elapsed: float) -> None:
    payload = {
        "placements": {str(k): float(v) for k, v in placements.items()},
        "elapsed_seconds": float(elapsed),
    }
    with open(_placements_path(method_dir), "w") as f:
        json.dump(payload, f)


def _load_placements(method_dir: str) -> Optional[Tuple[Dict[int, float], Optional[float]]]:
    p = _placements_path(method_dir)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        raw = json.load(f)
    # Back-compat: old format was a flat {tid: start} dict.
    if "placements" in raw:
        placements = {int(k): float(v) for k, v in raw["placements"].items()}
        elapsed = raw.get("elapsed_seconds")
        return placements, (float(elapsed) if elapsed is not None else None)
    return {int(k): float(v) for k, v in raw.items()}, None


def _parse_gpus(spec: str) -> List[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tubes_npz_dir", required=True,
                        help="Directory containing tube_*.npz files produced by TubeArchive.save_all.")
    parser.add_argument("--output_dir", default="./comparison_out",
                        help="Where to write per-method plots and the combined comparison plot.")
    parser.add_argument("--methods", default="energy,pso,mcts,mcmc",
                        help="Comma-separated subset of {energy,pso,mcts,mcmc}.")
    parser.add_argument("--video_length_frames", type=int, default=0,
                        help="Override the timeline length (frames). "
                             "Default: read metadata.json (augmented sets) or infer from tube frame indices.")

    parser.add_argument("--collision_method", default="centroid",
                        help="centroid (default, hinge on centroid distance) | "
                             "iou (true overlap only) | repulsion (smooth proximity tax everywhere).")
    parser.add_argument("--sigma", type=float, default=50.0,
                        help="Repulsion softness (only used by --collision_method repulsion).")
    parser.add_argument("--collision_radius", type=float, default=30.0,
                        help="Centroid-distance cutoff in pixels (only used by --collision_method centroid). "
                             "Pairs farther than this contribute zero collision cost.")
    parser.add_argument("--w_chronology", type=float, default=0.0,
                        help="Weight on chronology preservation (Nie et al. TIP 2019 Et term). "
                             "Penalises synopsis-time inversions of pairs that were close in source. "
                             "0 = disabled.")

    parser.add_argument("--mcmc_chains", type=int, default=32,
                        help="Number of parallel MCMC chains (each is a candidate placement).")
    parser.add_argument("--mcmc_steps", type=int, default=50000,
                        help="MCMC iterations per chain.")
    parser.add_argument("--mcmc_proposal_std", type=float, default=5.0,
                        help="Stddev of Gaussian proposal kernel in seconds.")
    parser.add_argument("--mcmc_global_jump_prob", type=float, default=0.1,
                        help="Probability of replacing the proposal with a uniform draw "
                             "across the valid range (helps escape local minima).")
    parser.add_argument("--mcmc_optimize_speed", action="store_true",
                        help="Let MCMC propose per-paragraph playback speed changes "
                             "(synopsis duration of paragraph k = src_dur[k] / speed[k]).")
    parser.add_argument("--mcmc_optimize_size", action="store_true",
                        help="Let MCMC propose per-paragraph spatial size scales "
                             "(bbox shrinks toward centroid; effective collision radius "
                             "scales with avg(size_i, size_j)).")
    parser.add_argument("--paragraph_seconds", type=float, default=2.0,
                        help="Source-time length of each paragraph (one speed/size knob per paragraph).")
    parser.add_argument("--speed_min", type=float, default=0.5)
    parser.add_argument("--speed_max", type=float, default=4.0)
    parser.add_argument("--size_min", type=float, default=0.5)
    parser.add_argument("--w_speed_reg", type=float, default=1.0,
                        help="Penalty weight on speed deviation from 1.0; without this the "
                             "optimizer tends to extreme speeds that look bad on playback.")
    parser.add_argument("--w_size_reg", type=float, default=1.0,
                        help="Penalty weight on size deviation from 1.0.")
    parser.add_argument("--fps", type=float, default=0.0,
                        help="FPS for converting video_length_frames -> seconds. "
                             "Default: read metadata.json or fall back to 30.")

    parser.add_argument("--energy_epochs", type=int, default=2000)

    parser.add_argument("--pso_num_particles", type=int, default=30)
    parser.add_argument("--pso_max_iterations", type=int, default=500)

    parser.add_argument("--mcts_training_episodes", type=int, default=2)
    parser.add_argument("--mcts_games_per_episode", type=int, default=4)
    parser.add_argument("--mcts_sims_training", type=int, default=100)
    parser.add_argument("--mcts_sims_final", type=int, default=300)

    parser.add_argument("--parallel", action="store_true",
                        help="Run each method as its own subprocess (one per GPU).")
    parser.add_argument("--gpus", default="",
                        help="Comma-separated GPU ids for --parallel (e.g. '0,1'). "
                             "Round-robin assigned to methods. Empty = no GPU pinning.")
    parser.add_argument("--single_method", action="store_true",
                        help=argparse.SUPPRESS)  # internal: writes placements.json then exits

    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    log = logging.getLogger("compare_optimizers")

    if not os.path.isdir(args.tubes_npz_dir):
        log.error(f"tubes_npz_dir does not exist: {args.tubes_npz_dir}")
        return 1

    log.info(f"Loading tubes from {args.tubes_npz_dir}")
    tubes = TubeArchive.load_all(args.tubes_npz_dir)
    if not tubes:
        log.error("No tubes loaded — nothing to compare.")
        return 1
    log.info(f"Loaded {len(tubes)} tubes")

    video_length, source = _resolve_video_length(args.tubes_npz_dir, tubes, args.video_length_frames)
    fps = _resolve_fps(args.tubes_npz_dir, args.fps)
    log.info(f"Video length: {video_length} frames @ {fps} fps "
             f"= {video_length / fps:.1f}s (source: {source})")

    os.makedirs(args.output_dir, exist_ok=True)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]

    # Single-method worker mode (used by --parallel parent): run, persist, exit.
    if args.single_method:
        if len(methods) != 1:
            log.error(f"--single_method expects exactly one method, got {methods}")
            return 1
        name = methods[0]
        placements, elapsed = _run_one_method(name, args, tubes, video_length, fps, log)
        method_dir = os.path.join(args.output_dir, name)
        _save_placements(method_dir, placements, elapsed)
        log.info(f"Wrote {_placements_path(method_dir)} (elapsed {elapsed:.1f}s)")
        return 0

    placements_per_method: Dict[str, Dict[int, float]] = {}
    times_per_method: Dict[str, float] = {}
    parallel_t0 = time.time()

    if args.parallel:
        gpu_ids = _parse_gpus(args.gpus) if args.gpus else []
        if gpu_ids:
            log.info(f"Parallel mode: queue-based, 1 method per GPU at a time over {gpu_ids}")
        else:
            log.info("Parallel mode: no GPUs configured — running sequentially "
                     "(use --gpus to enable parallelism, e.g. --gpus 0,1)")

        # Queue-based scheduler: never pack two methods onto the same GPU at
        # the same time. When all GPUs are busy, wait for any to free up.
        slots: List[Optional[int]] = list(gpu_ids) if gpu_ids else [None]
        pending: List[str] = list(methods)
        in_flight: Dict[str, Tuple[str, subprocess.Popen, float]] = {}
        # in_flight maps slot_key -> (method_name, popen, start_time)
        # slot_key is the str(gpu_id) (or "cpu") so we can free it when done.

        def _slot_key(g: Optional[int]) -> str:
            return "cpu" if g is None else str(g)

        free_slots: List[Optional[int]] = list(slots)
        failed: List[str] = []

        while pending or in_flight:
            # Dispatch any pending method onto any free slot.
            while pending and free_slots:
                gpu = free_slots.pop(0)
                method = pending.pop(0)
                key = _slot_key(gpu)
                proc = _spawn_subprocess(method, gpu, args, log)
                in_flight[key] = (method, proc, time.time())

            if not in_flight:
                break

            # Poll for the first finished subprocess.
            done = None
            while done is None:
                for key, (method, proc, t0) in in_flight.items():
                    ret = proc.poll()
                    if ret is not None:
                        done = (key, method, proc, t0, ret)
                        break
                if done is None:
                    time.sleep(0.5)

            key, method, proc, t0, ret = done
            elapsed = time.time() - t0
            del in_flight[key]
            free_slots.append(None if key == "cpu" else int(key))
            if ret != 0:
                log.error(f"{method} (gpu={key}) exited with code {ret} after {elapsed:.1f}s "
                          f"(see {os.path.join(args.output_dir, method, 'stdout.log')})")
                failed.append(method)
            else:
                log.info(f"{method} (gpu={key}) done in {elapsed:.1f}s; "
                         f"queue: {len(pending)} pending, {len(in_flight)} running")

        for name in methods:
            if name in failed:
                continue
            method_dir = os.path.join(args.output_dir, name)
            loaded = _load_placements(method_dir)
            if loaded is None:
                log.warning(f"{name}: placements.json missing, skipping in comparison plot")
                continue
            placements, elapsed = loaded
            pretty = _PRETTY_NAME.get(name, name)
            placements_per_method[pretty] = placements
            if elapsed is not None:
                times_per_method[pretty] = elapsed
    else:
        for name in methods:
            placements, elapsed = _run_one_method(name, args, tubes, video_length, fps, log)
            pretty = _PRETTY_NAME.get(name, name)
            placements_per_method[pretty] = placements
            times_per_method[pretty] = elapsed

    if not placements_per_method:
        log.error("No method produced placements — nothing to plot.")
        return 1

    log.info(f"All methods done in {time.time() - parallel_t0:.1f}s wall-clock")
    comparison_path = os.path.join(args.output_dir, "methods_comparison.png")
    save_methods_comparison(tubes, placements_per_method, comparison_path,
                            times_per_method=times_per_method)
    log.info(f"Done. Comparison plot: {comparison_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

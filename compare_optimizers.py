"""Run every tube-placement optimizer on the same tube set and produce one
side-by-side comparison plot.

Inputs are pre-generated tubes saved as .npz files (see TubeArchive.save_all
or `--use_npz` in the main pipeline) — this avoids re-running detection and
tracking for every optimizer.

Usage:
    python compare_optimizers.py --tubes_npz_dir ./tubes_npz \\
        --output_dir ./comparison_out \\
        --methods energy,pso,mcts

Outputs:
    <output_dir>/<method>_optimized_plot.png  — one per method (initial vs optimized)
    <output_dir>/methods_comparison.png       — single figure with all methods
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Optional, Tuple

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


def _build_optimizer(name: str, output_dir: str, args):
    name = name.lower()
    if name == "energy":
        from video_synopsis.optimization.energy import EnergyOptimizer
        return EnergyOptimizer(
            epochs=args.energy_epochs,
            collision_method=args.collision_method,
            sigma=args.sigma,
            output_dir=output_dir,
        )
    if name == "pso":
        from video_synopsis.optimization.pso import PSOOptimizer
        return PSOOptimizer(
            num_particles=args.pso_num_particles,
            max_iterations=args.pso_max_iterations,
            collision_method=args.collision_method,
            sigma=args.sigma,
            output_dir=output_dir,
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
            output_dir=output_dir,
        )
    raise ValueError(f"Unknown optimizer: {name!r} (expected energy|pso|mcts)")


_PRETTY_NAME = {
    "energy": "Energy (gradient)",
    "pso": "PSO",
    "mcts": "MCTS+AlphaZero",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tubes_npz_dir", required=True,
                        help="Directory containing tube_*.npz files produced by TubeArchive.save_all.")
    parser.add_argument("--output_dir", default="./comparison_out",
                        help="Where to write per-method plots and the combined comparison plot.")
    parser.add_argument("--methods", default="energy,pso,mcts",
                        help="Comma-separated subset of {energy,pso,mcts}.")
    parser.add_argument("--video_length_frames", type=int, default=0,
                        help="Override the timeline length (frames). "
                             "Default: read metadata.json (augmented sets) or infer from tube frame indices.")

    parser.add_argument("--collision_method", default="repulsion")
    parser.add_argument("--sigma", type=float, default=50.0)

    parser.add_argument("--energy_epochs", type=int, default=2000)

    parser.add_argument("--pso_num_particles", type=int, default=30)
    parser.add_argument("--pso_max_iterations", type=int, default=500)

    parser.add_argument("--mcts_training_episodes", type=int, default=2)
    parser.add_argument("--mcts_games_per_episode", type=int, default=4)
    parser.add_argument("--mcts_sims_training", type=int, default=100)
    parser.add_argument("--mcts_sims_final", type=int, default=300)

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
    log.info(f"Video length: {video_length} frames (source: {source})")

    os.makedirs(args.output_dir, exist_ok=True)
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    placements_per_method: Dict[str, Dict[int, float]] = {}

    for name in methods:
        pretty = _PRETTY_NAME.get(name, name)
        log.info(f"=== Running {pretty} ===")
        method_dir = os.path.join(args.output_dir, name)
        os.makedirs(method_dir, exist_ok=True)
        optimizer = _build_optimizer(name, method_dir, args)
        t0 = time.time()
        placements = optimizer.optimize(tubes, video_length)
        elapsed = time.time() - t0
        log.info(f"{pretty} finished in {elapsed:.1f}s with {len(placements)} placements")
        placements_per_method[pretty] = placements

    comparison_path = os.path.join(args.output_dir, "methods_comparison.png")
    save_methods_comparison(tubes, placements_per_method, comparison_path)
    log.info(f"Done. Comparison plot: {comparison_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

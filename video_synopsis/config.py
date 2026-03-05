"""Dataclass-based configuration replacing supplementary/our_args.py."""

import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml


@dataclass
class Config:
    """Video Synopsis pipeline configuration."""

    # Input
    video: str = "../src/all_rush_video.mp4"
    fps: int = 25
    batch_size: int = 8
    buff_size: int = 32

    # Segmentation
    segmenter: str = "rfdetr"  # "rfdetr", "people", or "fastsam"
    input_model: str = "Unet_2020-07-20"
    fastsam_model: str = "FastSAM-s.pt"
    rfdetr_variant: str = "base"  # "base" or "large"
    rfdetr_threshold: float = 0.5
    ext: str = ".png"
    dvalue: int = 9

    # Tracking
    tracker: str = "botsort"  # "botsort", "sort", or "sam3"
    sort_max_age: int = 3
    sort_min_hits: int = 3
    sort_iou_threshold: float = 0.3
    botsort_track_high_thresh: float = 0.6
    botsort_track_low_thresh: float = 0.1
    botsort_new_track_thresh: float = 0.7
    botsort_track_buffer: int = 30
    botsort_match_thresh: float = 0.8
    botsort_with_reid: bool = True

    # Optimization
    energy_optimization: bool = True
    optimizer: str = "mcts"  # "mcts", "energy", or "pso"
    epochs: int = 2000
    collision_method: str = "repulsion"
    sigma: float = 50.0

    # MCTS-specific
    mcts_training_episodes: int = 10
    mcts_games_per_episode: int = 10
    mcts_sims_training: int = 200
    mcts_sims_final: int = 1000
    mcts_c_puct: float = 1.4
    mcts_lr: float = 0.001

    # PSO-specific
    pso_num_particles: int = 30
    pso_max_iterations: int = 500
    pso_inertia: float = 0.7
    pso_cognitive: float = 1.5
    pso_social: float = 1.5

    # Paths
    output: str = "output"
    masks: str = "../masks"
    synopsis_frames: str = "../synopsis_frames"
    optimized_tubes_dir: str = "../optimized_tubes"
    bg_path: str = "../bg/background_img.png"
    files_csv_dir: str = "*/*.csv"

    # Storage
    use_npz: bool = True  # Use .npz instead of per-frame PNG/CSV

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_args(cls, args: Optional[list] = None) -> "Config":
        parser = argparse.ArgumentParser(description="Video Synopsis Pipeline")

        parser.add_argument("-v", "--video", type=str, default=cls.video)
        parser.add_argument("-f", "--fps", type=int, default=cls.fps)
        parser.add_argument("-bsz", "--batch_size", type=int, default=cls.batch_size)
        parser.add_argument("-b", "--buff_size", type=int, default=cls.buff_size)

        parser.add_argument("--segmenter", type=str, default=cls.segmenter,
                            choices=["rfdetr", "people", "fastsam"])
        parser.add_argument("-inmod", "--input_model", type=str, default=cls.input_model)
        parser.add_argument("--fastsam_model", type=str, default=cls.fastsam_model)
        parser.add_argument("--rfdetr_variant", type=str, default=cls.rfdetr_variant,
                            choices=["base", "large"])
        parser.add_argument("--rfdetr_threshold", type=float, default=cls.rfdetr_threshold)
        parser.add_argument("-e", "--ext", type=str, default=cls.ext)
        parser.add_argument("-cv", "--dvalue", type=int, default=cls.dvalue)

        parser.add_argument("--tracker", type=str, default=cls.tracker,
                            choices=["botsort", "sort", "sam3"])
        parser.add_argument("--sort_max_age", type=int, default=cls.sort_max_age)
        parser.add_argument("--sort_min_hits", type=int, default=cls.sort_min_hits)
        parser.add_argument("--sort_iou_threshold", type=float, default=cls.sort_iou_threshold)

        parser.add_argument("--botsort_track_high_thresh", type=float, default=cls.botsort_track_high_thresh)
        parser.add_argument("--botsort_track_low_thresh", type=float, default=cls.botsort_track_low_thresh)
        parser.add_argument("--botsort_new_track_thresh", type=float, default=cls.botsort_new_track_thresh)
        parser.add_argument("--botsort_track_buffer", type=int, default=cls.botsort_track_buffer)
        parser.add_argument("--botsort_match_thresh", type=float, default=cls.botsort_match_thresh)
        parser.add_argument("--botsort_with_reid", action="store_true", default=cls.botsort_with_reid)
        parser.add_argument("--botsort_no_reid", action="store_true")

        parser.add_argument("--energy_optimization", type=bool, default=cls.energy_optimization)
        parser.add_argument("--optimizer", type=str, default=cls.optimizer,
                            choices=["mcts", "energy", "pso"])
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument("--collision_method", type=str, default=cls.collision_method)
        parser.add_argument("--sigma", type=float, default=cls.sigma)

        parser.add_argument("--mcts_training_episodes", type=int, default=cls.mcts_training_episodes)
        parser.add_argument("--mcts_games_per_episode", type=int, default=cls.mcts_games_per_episode)
        parser.add_argument("--mcts_sims_training", type=int, default=cls.mcts_sims_training)
        parser.add_argument("--mcts_sims_final", type=int, default=cls.mcts_sims_final)
        parser.add_argument("--mcts_c_puct", type=float, default=cls.mcts_c_puct)
        parser.add_argument("--mcts_lr", type=float, default=cls.mcts_lr)

        parser.add_argument("--pso_num_particles", type=int, default=cls.pso_num_particles)
        parser.add_argument("--pso_max_iterations", type=int, default=cls.pso_max_iterations)
        parser.add_argument("--pso_inertia", type=float, default=cls.pso_inertia)
        parser.add_argument("--pso_cognitive", type=float, default=cls.pso_cognitive)
        parser.add_argument("--pso_social", type=float, default=cls.pso_social)

        parser.add_argument("--output", type=str, default=cls.output)
        parser.add_argument("--masks", type=str, default=cls.masks)
        parser.add_argument("--synopsis_frames", type=str, default=cls.synopsis_frames)
        parser.add_argument("--optimized_tubes_dir", type=str, default=cls.optimized_tubes_dir)
        parser.add_argument("--bg_path", type=str, default=cls.bg_path)
        parser.add_argument("--files_csv_dir", type=str, default=cls.files_csv_dir)

        parser.add_argument("--use_npz", action="store_true", default=cls.use_npz)
        parser.add_argument("--no_npz", action="store_true")

        parsed = parser.parse_args(args)
        d = vars(parsed)
        if d.pop("no_npz", False):
            d["use_npz"] = False
        if d.pop("botsort_no_reid", False):
            d["botsort_with_reid"] = False
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_legacy_dict(self) -> dict:
        """Convert to the legacy dict format expected by old code."""
        return {
            "buff_size": self.buff_size,
            "video": self.video,
            "input_model": self.input_model,
            "ext": self.ext,
            "dvalue": self.dvalue,
            "fps": self.fps,
            "batch_size": self.batch_size,
            "files_csv_dir": self.files_csv_dir,
            "optimized_tubes_dir": self.optimized_tubes_dir,
            "output": self.output,
            "masks": self.masks,
            "synopsis_frames": self.synopsis_frames,
            "energy_optimization": self.energy_optimization,
            "epochs": self.epochs,
            "bg_path": self.bg_path,
            "sigma": self.sigma,
            "collision_method": self.collision_method,
            "use_mcts": self.optimizer == "mcts",
            "mcts_epochs": self.mcts_sims_final,
        }

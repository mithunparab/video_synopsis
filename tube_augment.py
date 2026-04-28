"""Tube augmentation for synthetic long-form benchmarking.

Loads tubes saved by the main pipeline (.npz format), produces a larger
augmented set via mirror / translate / time-reverse / speed-scale, and
writes them back as .npz plus a metadata.json describing the synthetic
timeline. The augmented directory can then be fed into main.py via
--tubes_npz_dir to skip inference and run only optimizer + stitch.

Usage:
    python tube_augment.py \\
        --input_dir ./tubes_npz \\
        --output_dir ./tubes_npz_aug \\
        --bg_path ./bg/background_img.png \\
        --multiplier 10 --target_minutes 60 --fps 25
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
from typing import Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_synopsis.data.types import Tube, TubeFrame
from video_synopsis.data.tube_store import TubeArchive

log = logging.getLogger(__name__)


def mirror_tube(tube: Tube, frame_width: int) -> Tube:
    """Horizontal flip: bbox x-coords mirrored, mask + ROI flipped column-wise."""
    new_frames = []
    for f in tube.frames:
        img = f.image
        new_image = img[:, ::-1, :].copy() if img.ndim == 3 else img[:, ::-1].copy()
        new_mask = f.mask[:, ::-1].copy()
        x1, y1, x2, y2 = f.bbox
        new_bbox = np.array([frame_width - x2, y1, frame_width - x1, y2], dtype=np.float32)
        new_frames.append(TubeFrame(
            frame_index=f.frame_index,
            bbox=new_bbox,
            mask=new_mask,
            image=new_image,
            timestamp=f.timestamp,
        ))
    return Tube(tube_id=tube.tube_id, frames=new_frames)


def translate_tube(
    tube: Tube, dx: float, dy: float, frame_width: int, frame_height: int
) -> Tube:
    """Shift bboxes by (dx, dy), clamping the shift so the tube stays in-frame."""
    bboxes = tube.bboxes_array
    if bboxes.size == 0:
        return tube

    min_x1 = float(bboxes[:, 0].min())
    max_x2 = float(bboxes[:, 2].max())
    min_y1 = float(bboxes[:, 1].min())
    max_y2 = float(bboxes[:, 3].max())
    dx = float(np.clip(dx, -min_x1, frame_width - max_x2))
    dy = float(np.clip(dy, -min_y1, frame_height - max_y2))

    new_frames = []
    for f in tube.frames:
        x1, y1, x2, y2 = f.bbox
        new_bbox = np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy], dtype=np.float32)
        new_frames.append(TubeFrame(
            frame_index=f.frame_index,
            bbox=new_bbox,
            mask=f.mask.copy(),
            image=f.image.copy(),
            timestamp=f.timestamp,
        ))
    return Tube(tube_id=tube.tube_id, frames=new_frames)


def reverse_tube(tube: Tube) -> Tube:
    """Reverse the temporal order. Frames replay backwards on the same path."""
    if tube.num_frames == 0:
        return tube
    timestamps = tube.timestamps_array
    t_min, t_max = float(timestamps.min()), float(timestamps.max())
    new_frames = []
    for f in tube.frames:
        new_t = t_max - (f.timestamp - t_min)
        new_frames.append(TubeFrame(
            frame_index=f.frame_index,
            bbox=f.bbox.copy(),
            mask=f.mask.copy(),
            image=f.image.copy(),
            timestamp=new_t,
        ))
    new_frames.sort(key=lambda x: x.timestamp)
    return Tube(tube_id=tube.tube_id, frames=new_frames)


def speed_scale_tube(tube: Tube, scale: float) -> Tube:
    """Resample the tube at a different playback speed (>1 = faster, fewer frames)."""
    if tube.num_frames == 0 or scale <= 0:
        return tube

    if scale > 1.0:
        step = max(1, int(round(scale)))
        kept = tube.frames[::step]
    elif scale < 1.0:
        step = max(1, int(round(1.0 / scale)))
        kept = []
        for f in tube.frames:
            kept.extend([f] * step)
    else:
        kept = list(tube.frames)

    if not kept:
        return tube

    t0 = tube.start_time
    base_dt = (tube.end_time - t0) / max(len(tube.frames) - 1, 1)
    new_frames = []
    for i, f in enumerate(kept):
        new_frames.append(TubeFrame(
            frame_index=f.frame_index,
            bbox=f.bbox.copy(),
            mask=f.mask.copy(),
            image=f.image.copy(),
            timestamp=t0 + i * base_dt,
        ))
    return Tube(tube_id=tube.tube_id, frames=new_frames)


def augment_one(
    tube: Tube,
    new_id: int,
    frame_width: int,
    frame_height: int,
    mirror_prob: float,
    translate_sigma: Tuple[float, float],
    reverse_prob: float,
    speed_range: Tuple[float, float],
) -> Tube:
    out = tube
    if random.random() < mirror_prob:
        out = mirror_tube(out, frame_width)
    if translate_sigma[0] > 0 or translate_sigma[1] > 0:
        dx = random.gauss(0, translate_sigma[0])
        dy = random.gauss(0, translate_sigma[1])
        out = translate_tube(out, dx, dy, frame_width, frame_height)
    if random.random() < reverse_prob:
        out = reverse_tube(out)
    if speed_range[0] != 1.0 or speed_range[1] != 1.0:
        scale = random.uniform(*speed_range)
        out = speed_scale_tube(out, scale)
    out.tube_id = new_id
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment tubes for synthetic long-form benchmarking."
    )
    parser.add_argument("--input_dir", required=True, help="Directory with original tube_*.npz files.")
    parser.add_argument("--output_dir", required=True, help="Output directory for augmented tubes.")
    parser.add_argument("--bg_path", required=True, help="Background image (used for frame size).")
    parser.add_argument("--multiplier", type=int, default=10, help="Augmented copies per original tube.")
    parser.add_argument("--target_minutes", type=float, default=60.0, help="Synthetic timeline length in minutes.")
    parser.add_argument("--fps", type=int, default=25, help="FPS for the synthetic timeline.")
    parser.add_argument("--include_original", action="store_true", help="Also include the unmodified original tubes.")
    parser.add_argument("--mirror_prob", type=float, default=0.5)
    parser.add_argument("--translate_sigma_x", type=float, default=80.0,
                        help="Std-dev (px) of horizontal translation. Set 0 to disable.")
    parser.add_argument("--translate_sigma_y", type=float, default=20.0,
                        help="Std-dev (px) of vertical translation. Keep small to stay on ground plane.")
    parser.add_argument("--reverse_prob", type=float, default=0.3)
    parser.add_argument("--speed_min", type=float, default=0.85)
    parser.add_argument("--speed_max", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)

    bg = cv2.imread(args.bg_path)
    if bg is None:
        raise FileNotFoundError(f"Cannot read background: {args.bg_path}")
    frame_height, frame_width = bg.shape[:2]
    log.info(f"Frame size: {frame_width}x{frame_height}")

    tubes = TubeArchive.load_all(args.input_dir)
    if not tubes:
        raise RuntimeError(f"No tubes found in {args.input_dir}")
    log.info(f"Loaded {len(tubes)} original tubes")

    os.makedirs(args.output_dir, exist_ok=True)

    next_id = 1
    augmented = {}

    if args.include_original:
        for t in tubes.values():
            new = copy.deepcopy(t)
            new.tube_id = next_id
            augmented[next_id] = new
            next_id += 1

    for t in tubes.values():
        for _ in range(args.multiplier):
            new = augment_one(
                copy.deepcopy(t),
                new_id=next_id,
                frame_width=frame_width,
                frame_height=frame_height,
                mirror_prob=args.mirror_prob,
                translate_sigma=(args.translate_sigma_x, args.translate_sigma_y),
                reverse_prob=args.reverse_prob,
                speed_range=(args.speed_min, args.speed_max),
            )
            augmented[next_id] = new
            next_id += 1

    log.info(f"Generated {len(augmented)} augmented tubes")
    TubeArchive.save_all(augmented, args.output_dir)

    target_frames = int(args.target_minutes * 60.0 * args.fps)
    metadata = {
        "num_tubes": len(augmented),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": args.fps,
        "target_video_length_seconds": args.target_minutes * 60.0,
        "target_video_length_frames": target_frames,
        "bg_path": os.path.abspath(args.bg_path),
        "args": vars(args),
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Metadata saved to {meta_path}")
    log.info(
        f"Synthetic timeline: {args.target_minutes:.1f} min @ {args.fps} fps "
        f"= {target_frames} frames"
    )
    log.info(f"Now run: python main.py --tubes_npz_dir {args.output_dir} --optimizer <energy|mcts|pso>")


if __name__ == "__main__":
    main()

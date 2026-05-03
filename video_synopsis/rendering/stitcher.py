"""Tube compositing / stitching onto background.

Replaces tube_util.py with efficient in-memory rendering using Tube dataclass
and TubeArchive instead of per-frame PIL.Image.open().
"""

import logging
from collections import defaultdict
from typing import Dict, Optional

import cv2
import numpy as np

from video_synopsis.data.types import Tube
from video_synopsis.optimization.schedule import Schedule

log = logging.getLogger(__name__)


def blend_roi_on_background(
    background: np.ndarray,
    roi: np.ndarray,
    roi_mask: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> np.ndarray:
    """Alpha-blend an ROI onto a background using a mask.

    Args:
        background: BGR background image (modified in-place).
        roi: BGR ROI image.
        roi_mask: Grayscale mask for the ROI.
        x1, y1, x2, y2: Coordinates on the background.

    Returns:
        Modified background.
    """
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, background.shape[1]), min(y2, background.shape[0])

    if x2 <= x1 or y2 <= y1:
        return background

    roi_w, roi_h = x2 - x1, y2 - y1
    roi_resized = cv2.resize(roi, (roi_w, roi_h))
    mask_resized = cv2.resize(roi_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    _, mask_bin = cv2.threshold(mask_resized, 1, 255, cv2.THRESH_BINARY)
    if mask_bin.dtype != np.uint8:
        mask_bin = mask_bin.astype(np.uint8)

    region = background[y1:y2, x1:x2]
    if roi_resized.shape[:2] != region.shape[:2]:
        return background

    fg = cv2.bitwise_and(roi_resized, roi_resized, mask=mask_bin)
    bg = cv2.bitwise_and(region, region, mask=cv2.bitwise_not(mask_bin))
    background[y1:y2, x1:x2] = cv2.add(fg, bg)
    return background


class Stitcher:
    """Composites optimized tubes onto a background to produce the synopsis video."""

    def __init__(self, bgimg: np.ndarray, fps: int = 25):
        self.bgimg = bgimg
        self.fps = fps

    def render(
        self,
        tubes: Dict[int, Tube],
        optimized_starts: Dict[int, float],
        output_path: str,
    ) -> str:
        """Render tubes onto background and write output video.

        Args:
            tubes: Dict of tube_id -> Tube (with image/mask data).
            optimized_starts: Dict of tube_id -> optimized start time.
            output_path: Path to output video file.

        Returns:
            Path to the output video.
        """
        h, w = self.bgimg.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {output_path}")

        # Build timeline: map time -> list of (tube_id, frame_idx_in_tube)
        timeline: Dict[float, list] = defaultdict(list)

        for tid, tube in tubes.items():
            if tube.num_frames == 0 or tid not in optimized_starts:
                continue
            start_offset = optimized_starts[tid]
            min_ts = tube.start_time

            for i, frame in enumerate(tube.frames):
                new_time = (frame.timestamp - min_ts) + start_offset
                # Quantize to frame boundaries
                quantized = round(new_time * self.fps) / self.fps
                timeline[quantized].append((tid, i))

        if not timeline:
            log.warning("No frames to render. Writing empty video.")
            writer.release()
            return output_path

        sorted_times = sorted(timeline.keys())

        for t in sorted_times:
            bg = self.bgimg.copy()

            for tid, frame_idx in timeline[t]:
                tube = tubes[tid]
                tf = tube.frames[frame_idx]
                x1, y1, x2, y2 = tf.bbox.astype(int)
                blend_roi_on_background(bg, tf.image, tf.mask, x1, y1, x2, y2)

            writer.write(bg)

        writer.release()
        log.info(f"Synopsis video saved to {output_path} ({len(sorted_times)} frames)")
        return output_path

    def render_with_schedules(
        self,
        tubes: Dict[int, Tube],
        schedules: Dict[int, Schedule],
        output_path: str,
    ) -> str:
        """Render with full schedules (per-paragraph speed and size).

        For each synopsis frame ``t``, walk every tube whose schedule covers ``t``
        and look up the source frame via the paragraph's speed-warped time map.
        Each frame's bbox is shrunk toward its centroid by the paragraph's size
        factor, then composited onto the background.
        """
        h, w = self.bgimg.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {output_path}")

        synopsis_end = 0.0
        for sched in schedules.values():
            synopsis_end = max(synopsis_end, sched.start + sched.synopsis_duration)
        if synopsis_end <= 0:
            log.warning("Empty schedules — writing empty video.")
            writer.release()
            return output_path

        n_frames = int(round(synopsis_end * self.fps)) + 1
        log.info(f"Rendering {n_frames} synopsis frames @ {self.fps} fps")

        # Per-tube cumulative paragraph-start lookup, sorted source-time arrays
        per_tube_cum = {tid: s.syn_para_cum_starts for tid, s in schedules.items()}
        per_tube_src_ts = {}
        for tid, tube in tubes.items():
            if tid not in schedules:
                continue
            ts = tube.timestamps_array
            if ts.size == 0:
                continue
            per_tube_src_ts[tid] = ts - ts.min()

        for frame_i in range(n_frames):
            t_syn = frame_i / self.fps
            bg = self.bgimg.copy()

            for tid, sched in schedules.items():
                if tid not in tubes or tid not in per_tube_src_ts:
                    continue
                if t_syn < sched.start or t_syn >= sched.start + sched.synopsis_duration:
                    continue
                # Find which paragraph
                rel = t_syn - sched.start
                cum = per_tube_cum[tid]
                durs = sched.syn_para_durs
                K = sched.num_paragraphs
                # Linear scan is fine: K is typically 1–10.
                k = K - 1
                for kk in range(K):
                    if rel < cum[kk] + durs[kk]:
                        k = kk
                        break
                t_src = float(sched.src_para_starts[k] + (rel - cum[k]) * sched.speeds[k])
                size = float(sched.sizes[k])

                # Find nearest source frame
                src_ts_rel = per_tube_src_ts[tid]
                idx = int(np.argmin(np.abs(src_ts_rel - t_src)))
                tf = tubes[tid].frames[idx]
                bbox = tf.bbox.astype(np.float32)
                # Shrink bbox toward centroid by ``size``
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                hw = (bbox[2] - bbox[0]) / 2 * size
                hh = (bbox[3] - bbox[1]) / 2 * size
                x1 = int(cx - hw); y1 = int(cy - hh)
                x2 = int(cx + hw); y2 = int(cy + hh)
                blend_roi_on_background(bg, tf.image, tf.mask, x1, y1, x2, y2)

            writer.write(bg)

        writer.release()
        log.info(f"Synopsis video saved to {output_path} ({n_frames} frames)")
        return output_path

    def render_legacy(
        self,
        tubes: Dict[int, Tube],
        optimized_starts: Dict[int, float],
        video_writer: cv2.VideoWriter,
    ) -> None:
        """Render directly to an existing cv2.VideoWriter (for backward compat)."""
        timeline: Dict[float, list] = defaultdict(list)

        for tid, tube in tubes.items():
            if tube.num_frames == 0 or tid not in optimized_starts:
                continue
            start_offset = optimized_starts[tid]
            min_ts = tube.start_time

            for i, frame in enumerate(tube.frames):
                new_time = (frame.timestamp - min_ts) + start_offset
                quantized = round(new_time * self.fps) / self.fps
                timeline[quantized].append((tid, i))

        sorted_times = sorted(timeline.keys())

        for t in sorted_times:
            bg = self.bgimg.copy()
            for tid, frame_idx in timeline[t]:
                tube = tubes[tid]
                tf = tube.frames[frame_idx]
                x1, y1, x2, y2 = tf.bbox.astype(int)
                blend_roi_on_background(bg, tf.image, tf.mask, x1, y1, x2, y2)
            video_writer.write(bg)

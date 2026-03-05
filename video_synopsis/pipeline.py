"""Main pipeline orchestrator for video synopsis."""

import datetime
import logging
import os
import shutil
import time
from typing import Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

from video_synopsis.config import Config
from video_synopsis.data.types import Tube, TubeFrame
from video_synopsis.data.tube_store import TubeArchive
from video_synopsis.models.base import BaseSegmenter, BaseTracker
from video_synopsis.optimization.base import BaseOptimizer
from video_synopsis.rendering.background import extract_background
from video_synopsis.rendering.stitcher import Stitcher

log = logging.getLogger(__name__)


class Pipeline:
    """End-to-end video synopsis pipeline.

    Orchestrates: segmentation -> tracking -> tube generation -> optimization -> stitching.
    """

    def __init__(self, config: Config):
        self.config = config
        self._segmenter: Optional[BaseSegmenter] = None
        self._tracker: Optional[BaseTracker] = None
        self._optimizer: Optional[BaseOptimizer] = None

    @property
    def segmenter(self) -> BaseSegmenter:
        if self._segmenter is None:
            self._segmenter = self._create_segmenter()
        return self._segmenter

    @segmenter.setter
    def segmenter(self, val: BaseSegmenter) -> None:
        self._segmenter = val

    @property
    def tracker(self) -> BaseTracker:
        if self._tracker is None:
            self._tracker = self._create_tracker()
        return self._tracker

    @tracker.setter
    def tracker(self, val: BaseTracker) -> None:
        self._tracker = val

    @property
    def optimizer(self) -> BaseOptimizer:
        if self._optimizer is None:
            self._optimizer = self._create_optimizer()
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val: BaseOptimizer) -> None:
        self._optimizer = val

    def _create_single_segmenter(self, device: str = "") -> BaseSegmenter:
        """Create a segmenter instance on a specific device."""
        cfg = self.config
        if cfg.segmenter == "rfdetr":
            from video_synopsis.models.segmenters.rfdetr import RFDETRSegmenter
            return RFDETRSegmenter(
                model_variant=cfg.rfdetr_variant,
                threshold=cfg.rfdetr_threshold,
                device=device,
            )
        elif cfg.segmenter == "fastsam":
            from video_synopsis.models.segmenters.fastsam import FastSAMSegmenter
            return FastSAMSegmenter(model_path=cfg.fastsam_model, device=device)
        else:
            from video_synopsis.models.segmenters.people_seg import PeopleSegmenter
            return PeopleSegmenter(
                model_name=cfg.input_model,
                batch_size=cfg.batch_size,
                device=device,
            )

    def _resolve_num_gpus(self) -> int:
        """Determine how many GPUs to use."""
        if self.config.num_gpus > 0:
            return self.config.num_gpus
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        return 1

    def _create_segmenter(self) -> BaseSegmenter:
        num_gpus = self._resolve_num_gpus()
        if num_gpus > 1:
            from video_synopsis.models.segmenters.multi_gpu import MultiGPUSegmenter
            log.info(f"Creating multi-GPU segmenter with {num_gpus} GPUs")
            return MultiGPUSegmenter(
                factory_fn=self._create_single_segmenter,
                gpu_ids=list(range(num_gpus)),
            )
        return self._create_single_segmenter(self.config.device)

    def _create_tracker(self) -> BaseTracker:
        cfg = self.config
        if cfg.tracker == "botsort":
            from video_synopsis.models.trackers.botsort_tracker import BoTSORTTracker
            return BoTSORTTracker(
                track_high_thresh=cfg.botsort_track_high_thresh,
                track_low_thresh=cfg.botsort_track_low_thresh,
                new_track_thresh=cfg.botsort_new_track_thresh,
                track_buffer=cfg.botsort_track_buffer,
                match_thresh=cfg.botsort_match_thresh,
                with_reid=cfg.botsort_with_reid,
                device=cfg.device,
            )
        elif cfg.tracker == "sam3":
            from video_synopsis.models.trackers.sam3_tracker import SAM3Tracker
            return SAM3Tracker()
        else:
            from video_synopsis.models.trackers.sort_tracker import SORTTracker
            return SORTTracker(
                max_age=cfg.sort_max_age,
                min_hits=cfg.sort_min_hits,
                iou_threshold=cfg.sort_iou_threshold,
            )

    def _create_optimizer(self) -> BaseOptimizer:
        cfg = self.config
        if cfg.optimizer == "energy":
            from video_synopsis.optimization.energy import EnergyOptimizer
            return EnergyOptimizer(
                epochs=cfg.epochs,
                collision_method=cfg.collision_method,
                sigma=cfg.sigma,
            )
        elif cfg.optimizer == "pso":
            from video_synopsis.optimization.pso import PSOOptimizer
            return PSOOptimizer(
                num_particles=cfg.pso_num_particles,
                max_iterations=cfg.pso_max_iterations,
                w=cfg.pso_inertia,
                c1=cfg.pso_cognitive,
                c2=cfg.pso_social,
                collision_method=cfg.collision_method,
                sigma=cfg.sigma,
            )
        else:
            from video_synopsis.optimization.mcts import MCTSOptimizer
            return MCTSOptimizer(
                num_training_episodes=cfg.mcts_training_episodes,
                games_per_episode=cfg.mcts_games_per_episode,
                mcts_sims_training=cfg.mcts_sims_training,
                mcts_sims_final=cfg.mcts_sims_final,
                lr=cfg.mcts_lr,
                collision_method=cfg.collision_method,
                sigma=cfg.sigma,
                c_puct=cfg.mcts_c_puct,
                output_dir=cfg.optimized_tubes_dir,
            )

    def run(self, video_path: Optional[str] = None) -> str:
        """Run the full pipeline.

        Args:
            video_path: Path to input video. Uses config.video if not provided.

        Returns:
            Path to the output synopsis video.
        """
        cfg = self.config
        video_path = video_path or cfg.video
        start_time = time.time()

        log.info(f"Starting Video Synopsis pipeline for: {video_path}")

        # Setup directories
        self._setup_dirs()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or cfg.fps
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        log.info(f"Video: {frame_width}x{frame_height}, {video_length} frames, {fps} fps")

        # Extract background
        bgimg = extract_background(video_path, num_samples=fps, save_path=cfg.bg_path)
        log.info("Background extracted")

        # Generate tubes
        tubes = self._generate_tubes(cap, video_length, fps, frame_width, frame_height, bgimg)
        cap.release()
        log.info(f"Generated {len(tubes)} tubes")

        if not tubes:
            log.warning("No tubes generated. Cannot create synopsis.")
            return ""

        # Save tubes
        if cfg.use_npz:
            tubes_dir = os.path.join(os.path.dirname(cfg.output), "tubes_npz")
            TubeArchive.save_all(tubes, tubes_dir)
            log.info(f"Tubes saved to {tubes_dir}")

        # Optimize
        optimized_starts: Dict[int, float] = {}
        if cfg.energy_optimization:
            log.info(f"Optimizing with {cfg.optimizer}...")
            optimized_starts = self.optimizer.optimize(tubes, video_length)
        else:
            # Use original timestamps
            for tid, tube in tubes.items():
                optimized_starts[tid] = tube.start_time

        # Render
        output_video = os.path.abspath(
            os.path.join(
                os.path.dirname(cfg.output),
                f"{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.mp4",
            )
        )
        stitcher = Stitcher(bgimg, fps=fps)
        stitcher.render(tubes, optimized_starts, output_video)

        # Log timing summary
        elapsed = time.time() - start_time
        orig_duration = video_length / fps
        synopsis_duration = max(
            (optimized_starts[tid] + tubes[tid].duration) for tid in tubes
        )
        orig_m, orig_s = divmod(int(orig_duration), 60)
        syn_m, syn_s = divmod(int(synopsis_duration), 60)
        log.info(
            f"Original video: {orig_m}m {orig_s}s | "
            f"Synopsis: {syn_m}m {syn_s}s | "
            f"Compression: {orig_duration / synopsis_duration:.1f}x | "
            f"Pipeline time: {elapsed:.1f}s"
        )
        log.info(f"Output: {output_video}")

        return output_video

    def _setup_dirs(self) -> None:
        cfg = self.config
        for path in [cfg.output, cfg.synopsis_frames, cfg.masks]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def _is_moving(frame: np.ndarray, bg: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   motion_thresh: int = 30, motion_ratio: float = 0.15) -> bool:
        """Check if the bbox region differs enough from background to be considered moving."""
        roi_frame = frame[y1:y2, x1:x2]
        roi_bg = bg[y1:y2, x1:x2]
        if roi_frame.shape != roi_bg.shape:
            roi_bg = cv2.resize(roi_bg, (roi_frame.shape[1], roi_frame.shape[0]))
        diff = cv2.absdiff(roi_frame, roi_bg)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
        moving_pixels = np.count_nonzero(gray_diff > motion_thresh)
        total_pixels = gray_diff.size
        return (moving_pixels / total_pixels) > motion_ratio if total_pixels > 0 else False

    def _generate_tubes(
        self,
        cap: cv2.VideoCapture,
        video_length: int,
        fps: int,
        width: int,
        height: int,
        bgimg: np.ndarray = None,
    ) -> Dict[int, Tube]:
        """Run segmentation + tracking to generate tubes."""
        cfg = self.config
        tubes: Dict[int, Tube] = {}
        next_id = 1
        id_mapping: Dict[int, int] = {}
        frame_index = 0

        pbar = tqdm(
            total=video_length // cfg.batch_size,
            desc="Processing batches",
            unit="batch",
        )

        while True:
            batch_frames = []
            batch_indices = []

            for _ in range(cfg.batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_indices.append(frame_index)
                frame_index += 1

            if not batch_frames:
                break

            # Segment
            seg_results = self.segmenter.segment_batch(batch_frames)

            # Track and build tubes
            for i, (seg, orig_frame) in enumerate(zip(seg_results, batch_frames)):
                current_time = batch_indices[i] / fps

                # Filter to moving objects only (skip static/parked)
                if seg.bboxes and bgimg is not None:
                    moving_bboxes = []
                    for bbox in seg.bboxes:
                        bx1, by1, bx2, by2 = bbox.astype(int)
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(width, bx2), min(height, by2)
                        if self._is_moving(orig_frame, bgimg, bx1, by1, bx2, by2):
                            moving_bboxes.append(bbox)
                    dets = np.array(moving_bboxes) if moving_bboxes else np.empty((0, 4))
                elif seg.bboxes:
                    dets = np.array(seg.bboxes)
                else:
                    dets = np.empty((0, 4))

                tracks = self.tracker.update(dets, frame=orig_frame)

                for track in tracks:
                    track_id = int(track[4])
                    coords = [max(0, int(c)) for c in track[:4]]
                    x1, y1, x2, y2 = coords

                    if track_id not in id_mapping:
                        id_mapping[track_id] = next_id
                        next_id += 1

                    tube_id = id_mapping[track_id]

                    # Extract ROI and mask
                    roi = orig_frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    # Find matching segmentation mask for this detection
                    mask_roi = self._find_matching_mask(
                        seg, x1, y1, x2, y2, orig_frame.shape[:2]
                    )

                    if tube_id not in tubes:
                        tubes[tube_id] = Tube(tube_id=tube_id)

                    tubes[tube_id].add_frame(TubeFrame(
                        frame_index=batch_indices[i],
                        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                        mask=mask_roi,
                        image=roi,
                        timestamp=current_time,
                    ))

            pbar.update(1)
            if frame_index > video_length:
                break

        pbar.close()
        return tubes

    def _find_matching_mask(
        self,
        seg_result,
        x1: int, y1: int, x2: int, y2: int,
        frame_shape: tuple,
    ) -> np.ndarray:
        """Find the segmentation mask that best matches the tracked bbox."""
        h, w = frame_shape
        best_iou = 0.0
        best_mask = None

        for seg_bbox, seg_mask in zip(seg_result.bboxes, seg_result.masks):
            sx1, sy1, sx2, sy2 = seg_bbox.astype(int)
            # Compute IoU between track bbox and seg bbox
            ix1 = max(x1, sx1)
            iy1 = max(y1, sy1)
            ix2 = min(x2, sx2)
            iy2 = min(y2, sy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (sx2 - sx1) * (sy2 - sy1)
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                # Create full-frame mask then crop to track bbox
                full_mask = np.zeros((h, w), dtype=np.uint8)
                mh, mw = seg_mask.shape[:2]
                sh, sw = sy2 - sy1, sx2 - sx1
                if mh != sh or mw != sw:
                    seg_mask = cv2.resize(seg_mask, (sw, sh))
                full_mask[sy1:sy2, sx1:sx2] = seg_mask
                best_mask = full_mask[y1:y2, x1:x2]

        if best_mask is None:
            # Fallback: create a full mask for the ROI
            best_mask = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)

        return best_mask

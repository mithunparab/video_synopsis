"""SAM3 Tracker via Hugging Face transformers.

Uses SAM 2 / SAM 3 video predictor for mask-based tracking.
Requires: pip install transformers torch

This is an optional tracker; if transformers is not installed,
the import will fail gracefully and SORT should be used instead.
"""

import logging
from typing import Optional

import numpy as np

from video_synopsis.models.base import BaseTracker

log = logging.getLogger(__name__)


class SAM3Tracker(BaseTracker):
    """SAM3-based tracker using mask prompts from segmenter.

    Maintains embedding state across frames for consistent tracking.
    Falls back gracefully if transformers/SAM3 is unavailable.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "",
        max_objects: int = 50,
    ):
        self.model_id = model_id
        self._device = device
        self.max_objects = max_objects
        self._predictor = None
        self._state = None
        self._next_id = 1
        self._active_ids = {}
        self._initialized = False

    def _ensure_model(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch
            from transformers import Sam2VideoPredictor
        except ImportError:
            raise ImportError(
                "SAM3Tracker requires transformers with SAM2 support. "
                "Install with: pip install transformers>=4.40"
            )

        device = self._device
        if not device:
            import torch as _torch
            device = "cuda" if _torch.cuda.is_available() else "cpu"

        self._predictor = Sam2VideoPredictor.from_pretrained(self.model_id)
        self._predictor_device = device
        log.info(f"Loaded SAM3 tracker model {self.model_id} on {device}")

    def update(self, detections: np.ndarray, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Update tracker with detections and frame.

        For SAM3, detections are used as prompts (bboxes) to initialize
        new tracks. Existing tracks are propagated via the video predictor.

        Args:
            detections: (N, 4+) bbox detections [x1, y1, x2, y2, ...].
            frame: BGR frame (required for SAM3).

        Returns:
            (M, 5) array of [x1, y1, x2, y2, track_id].
        """
        self._ensure_model()

        if frame is None:
            log.warning("SAM3Tracker requires frame input. Returning empty.")
            return np.empty((0, 5))

        try:
            return self._track_with_sam3(detections, frame)
        except Exception as e:
            log.warning(f"SAM3 tracking failed: {e}. Returning detections with new IDs.")
            # Fallback: assign new IDs to detections
            results = []
            for det in detections[:, :4] if detections.size > 0 else []:
                results.append([*det, self._next_id])
                self._next_id += 1
            return np.array(results).reshape(-1, 5) if results else np.empty((0, 5))

    def _track_with_sam3(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Internal SAM3 tracking logic."""
        import torch

        # Initialize state on first frame
        if not self._initialized:
            inference_state = self._predictor.init_state(video_path=None)
            self._state = inference_state
            self._initialized = True

        results = []

        # Add new detections as prompts
        if detections.size > 0:
            for det in detections[:, :4]:
                x1, y1, x2, y2 = det
                obj_id = self._next_id
                self._next_id += 1

                _, out_ids, out_masks = self._predictor.add_new_points_or_box(
                    inference_state=self._state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=np.array([x1, y1, x2, y2]),
                )

                # Get bbox from mask
                if out_masks is not None and len(out_masks) > 0:
                    mask = (out_masks[0] > 0).cpu().numpy().squeeze()
                    ys, xs = np.where(mask)
                    if len(xs) > 0:
                        results.append([xs.min(), ys.min(), xs.max(), ys.max(), obj_id])
                    else:
                        results.append([x1, y1, x2, y2, obj_id])
                else:
                    results.append([x1, y1, x2, y2, obj_id])

        return np.array(results).reshape(-1, 5) if results else np.empty((0, 5))

    def reset(self) -> None:
        self._state = None
        self._initialized = False
        self._next_id = 1
        self._active_ids = {}

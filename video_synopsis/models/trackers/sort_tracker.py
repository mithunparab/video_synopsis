"""SORT tracker wrapper using the existing sort.py implementation."""

import logging
import sys
import os
from typing import Optional

import numpy as np

from video_synopsis.models.base import BaseTracker

log = logging.getLogger(__name__)


class SORTTracker(BaseTracker):
    """Simple Online Realtime Tracker (SORT) wrapper.

    Wraps the existing sort.py KalmanFilter-based tracker.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracker = None
        self._ensure_tracker()

    def _ensure_tracker(self) -> None:
        # Import from the root-level sort.py
        # Add parent directory to path if needed
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from sort import Sort
        self._tracker = Sort(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
        )

    def update(self, detections: np.ndarray, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Update tracker with detections.

        Args:
            detections: (N, 4+) array of [x1, y1, x2, y2, ...].
            frame: Ignored by SORT.

        Returns:
            (M, 5) array of [x1, y1, x2, y2, track_id].
        """
        if detections.size == 0:
            return self._tracker.update(np.empty((0, 5)))
        # SORT expects (N, 5) with scores; add dummy score if needed
        if detections.shape[1] == 4:
            scores = np.ones((detections.shape[0], 1))
            detections = np.hstack([detections, scores])
        return self._tracker.update(detections)

    def reset(self) -> None:
        self._ensure_tracker()

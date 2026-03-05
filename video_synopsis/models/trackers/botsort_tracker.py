"""BoT-SORT tracker via boxmot — MOT leaderboard #1 with ReID support."""

import logging
from typing import Optional

import numpy as np

from video_synopsis.models.base import BaseTracker

log = logging.getLogger(__name__)


class BoTSORTTracker(BaseTracker):
    """BoT-SORT multi-object tracker wrapper.

    BoT-SORT combines Kalman filter, camera-motion compensation, and
    ReID feature matching for robust tracking through occlusions.

    Requires: pip install boxmot
    """

    def __init__(
        self,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        with_reid: bool = True,
        device: str = "",
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.with_reid = with_reid
        self._device = device
        self._tracker = None

    def _get_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_tracker(self) -> None:
        if self._tracker is not None:
            return
        try:
            from boxmot import BotSort
        except ImportError:
            try:
                from boxmot import BoTSORT as BotSort
            except ImportError:
                raise ImportError(
                    "BoT-SORT requires the boxmot package. "
                    "Install with: pip install boxmot"
                )

        device = self._get_device()
        self._tracker = BotSort(
            reid_weights=None,
            device=device,
            half=False,
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            with_reid=self.with_reid,
        )
        log.info(
            f"Loaded BoT-SORT tracker (device={device}, reid={self.with_reid})"
        )

    def update(
        self, detections: np.ndarray, frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Update tracker with detections.

        Args:
            detections: (N, 4+) array of [x1, y1, x2, y2, ...].
            frame: BGR frame — used for ReID feature extraction.

        Returns:
            (M, 5) array of [x1, y1, x2, y2, track_id].
        """
        self._ensure_tracker()

        if detections.size == 0:
            dets = np.empty((0, 6))
        else:
            # boxmot expects (N, 6): [x1, y1, x2, y2, conf, cls]
            n = detections.shape[0]
            if detections.shape[1] == 4:
                conf = np.ones((n, 1))
                cls = np.zeros((n, 1))
                dets = np.hstack([detections, conf, cls])
            elif detections.shape[1] == 5:
                cls = np.zeros((n, 1))
                dets = np.hstack([detections, cls])
            else:
                dets = detections[:, :6]

        tracks = self._tracker.update(dets, frame)

        if tracks is None or len(tracks) == 0:
            return np.empty((0, 5))

        # boxmot returns (M, 7+): [x1, y1, x2, y2, id, conf, cls, ...]
        # We need (M, 5): [x1, y1, x2, y2, id]
        result = np.empty((len(tracks), 5))
        result[:, :4] = tracks[:, :4]
        result[:, 4] = tracks[:, 4]
        return result

    def reset(self) -> None:
        self._tracker = None

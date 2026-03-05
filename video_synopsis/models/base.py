from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SegmentationResult:
    """Result from a segmenter for a single frame."""
    masks: List[np.ndarray]       # list of per-instance binary masks (H, W), uint8
    bboxes: List[np.ndarray]      # list of [x1, y1, x2, y2] arrays
    scores: List[float] = field(default_factory=list)


class BaseSegmenter(ABC):
    """Abstract base for instance segmentation models."""

    @abstractmethod
    def segment_batch(self, frames: List[np.ndarray]) -> List[SegmentationResult]:
        """Segment a batch of BGR frames.

        Args:
            frames: List of BGR numpy arrays.

        Returns:
            List of SegmentationResult, one per frame.
        """

    def segment(self, frame: np.ndarray) -> SegmentationResult:
        """Segment a single frame (convenience wrapper)."""
        return self.segment_batch([frame])[0]


class BaseTracker(ABC):
    """Abstract base for multi-object trackers."""

    @abstractmethod
    def update(self, detections: np.ndarray, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Update tracker with new detections.

        Args:
            detections: (N, 4+) array of [x1, y1, x2, y2, ...] detections.
            frame: Optional BGR frame (needed by some trackers like SAM3).

        Returns:
            (M, 5) array of [x1, y1, x2, y2, track_id] for active tracks.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""

from abc import ABC, abstractmethod
from typing import Dict

from video_synopsis.data.types import Tube


class BaseOptimizer(ABC):
    """Abstract base for tube placement optimizers."""

    @abstractmethod
    def optimize(self, tubes: Dict[int, Tube], video_length_frames: int) -> Dict[int, float]:
        """Compute optimized start times for each tube.

        Args:
            tubes: Mapping of tube_id -> Tube.
            video_length_frames: Total frames in original video.

        Returns:
            Mapping of tube_id -> optimized start time (in seconds or frames,
            depending on implementation).
        """

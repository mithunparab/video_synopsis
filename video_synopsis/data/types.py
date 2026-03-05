from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class TubeFrame:
    """A single frame of a tracked object tube."""
    frame_index: int
    bbox: np.ndarray  # shape (4,) — [x1, y1, x2, y2]
    mask: np.ndarray   # 2D uint8 mask for this crop region
    image: np.ndarray  # BGR crop of the object
    timestamp: float   # seconds into the original video

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return float(self.width * self.height)


@dataclass
class Tube:
    """A complete tracked-object tube: a sequence of per-frame observations."""
    tube_id: int
    frames: List[TubeFrame] = field(default_factory=list)

    def add_frame(self, frame: TubeFrame) -> None:
        self.frames.append(frame)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def bboxes_array(self) -> np.ndarray:
        """(N, 4) array of per-frame bounding boxes."""
        if not self.frames:
            return np.empty((0, 4), dtype=np.float32)
        return np.array([f.bbox for f in self.frames], dtype=np.float32)

    @property
    def timestamps_array(self) -> np.ndarray:
        """(N,) array of per-frame timestamps."""
        if not self.frames:
            return np.empty((0,), dtype=np.float64)
        return np.array([f.timestamp for f in self.frames], dtype=np.float64)

    @property
    def frame_indices_array(self) -> np.ndarray:
        """(N,) array of original frame indices."""
        if not self.frames:
            return np.empty((0,), dtype=np.int64)
        return np.array([f.frame_index for f in self.frames], dtype=np.int64)

    @property
    def mask_areas(self) -> np.ndarray:
        """(N,) array of mask pixel counts per frame."""
        if not self.frames:
            return np.empty((0,), dtype=np.float64)
        return np.array([float(np.count_nonzero(f.mask)) for f in self.frames], dtype=np.float64)

    @property
    def duration(self) -> float:
        """Duration in seconds (max timestamp - min timestamp)."""
        ts = self.timestamps_array
        if ts.size == 0:
            return 0.0
        return float(ts.max() - ts.min())

    @property
    def start_time(self) -> float:
        ts = self.timestamps_array
        return float(ts.min()) if ts.size > 0 else 0.0

    @property
    def end_time(self) -> float:
        ts = self.timestamps_array
        return float(ts.max()) if ts.size > 0 else 0.0

    def bbox_at_time(self, t: float) -> Optional[np.ndarray]:
        """Get the bbox closest to timestamp t via nearest-neighbor lookup."""
        ts = self.timestamps_array
        if ts.size == 0:
            return None
        idx = int(np.argmin(np.abs(ts - t)))
        return self.frames[idx].bbox.copy()

    def union_bbox(self) -> np.ndarray:
        """Static union bbox across all frames — use sparingly (causes the bug we're fixing)."""
        bboxes = self.bboxes_array
        if bboxes.size == 0:
            return np.zeros(4, dtype=np.float32)
        return np.array([
            bboxes[:, 0].min(),
            bboxes[:, 1].min(),
            bboxes[:, 2].max(),
            bboxes[:, 3].max(),
        ], dtype=np.float32)

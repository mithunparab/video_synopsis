"""Per-tube placement decisions: start time, plus optional per-paragraph
playback speed and spatial size scale.

A tube is split into K paragraphs of (roughly) equal source-time duration.
Each paragraph carries:
  * ``speed[k]`` — playback rate. ``speed=1`` is native; ``speed=2`` halves
    the synopsis duration of the paragraph; ``speed=0.5`` doubles it.
  * ``size[k]`` — spatial scale toward the centroid. ``size=1`` is full;
    ``size=0.5`` shrinks each frame's bbox to half its width and height.

In Tetris terms: speed scales the piece along the time axis, size scales it
in the (x, y) plane. The optimiser packs scaled pieces into the synopsis
volume; the renderer applies the same scales.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import numpy as np


def _split_paragraphs(duration: float, paragraph_seconds: float) -> int:
    """Number of paragraphs needed to cover ``duration`` at ~paragraph_seconds each."""
    if duration <= 0 or paragraph_seconds <= 0:
        return 1
    return max(1, int(math.ceil(duration / paragraph_seconds)))


@dataclass
class Schedule:
    """The optimiser's full decision for one tube.

    Attributes:
        tube_id: Tube identifier.
        start: Synopsis-time at which this tube's first rendered frame plays.
        src_para_starts: Source-time offset (relative to tube ``t0``) of each
            paragraph's first source frame, shape ``[K]``.
        src_para_durs: Source-time duration of each paragraph, shape ``[K]``.
            ``sum(src_para_durs) == tube.duration`` (in source seconds).
        speeds: Playback speed per paragraph, shape ``[K]``.
        sizes: Spatial scale per paragraph, shape ``[K]``.
    """
    tube_id: int
    start: float
    src_para_starts: np.ndarray
    src_para_durs: np.ndarray
    speeds: np.ndarray
    sizes: np.ndarray

    @property
    def num_paragraphs(self) -> int:
        return int(self.src_para_durs.shape[0])

    @property
    def synopsis_duration(self) -> float:
        """Total synopsis duration after speed scaling."""
        return float((self.src_para_durs / np.maximum(self.speeds, 1e-6)).sum())

    @property
    def syn_para_durs(self) -> np.ndarray:
        """Per-paragraph synopsis duration."""
        return self.src_para_durs / np.maximum(self.speeds, 1e-6)

    @property
    def syn_para_cum_starts(self) -> np.ndarray:
        """Per-paragraph synopsis start (relative to tube's ``start``).

        Element ``k`` is the synopsis offset at which paragraph ``k`` begins.
        """
        durs = self.syn_para_durs
        if durs.size == 0:
            return np.zeros((0,), dtype=durs.dtype)
        out = np.zeros_like(durs)
        out[1:] = np.cumsum(durs[:-1])
        return out

    def synopsis_to_source(self, t_syn: float):
        """Map a synopsis-time to ``(paragraph_idx, source_offset, size_at_that_time)``.

        Returns ``None`` if ``t_syn`` falls outside ``[start, start + synopsis_duration]``.
        """
        rel = t_syn - self.start
        if rel < 0:
            return None
        cum = self.syn_para_cum_starts
        durs = self.syn_para_durs
        for k in range(self.num_paragraphs):
            if rel < cum[k] + durs[k]:
                t_src = float(self.src_para_starts[k] + (rel - cum[k]) * self.speeds[k])
                return k, t_src, float(self.sizes[k])
        return None


def make_default_schedule(tube_id: int, tube_duration: float, start: float = 0.0,
                          paragraph_seconds: float = 2.0) -> Schedule:
    """Create a Schedule with native speed (1.0) and full size (1.0) everywhere."""
    K = _split_paragraphs(tube_duration, paragraph_seconds)
    base_dur = tube_duration / K if K > 0 else tube_duration
    src_para_starts = np.arange(K, dtype=np.float32) * base_dur
    src_para_durs = np.full((K,), base_dur, dtype=np.float32)
    # Last paragraph absorbs any rounding.
    if K > 0:
        src_para_durs[-1] = tube_duration - src_para_starts[-1]
    return Schedule(
        tube_id=int(tube_id),
        start=float(start),
        src_para_starts=src_para_starts,
        src_para_durs=src_para_durs,
        speeds=np.ones((K,), dtype=np.float32),
        sizes=np.ones((K,), dtype=np.float32),
    )


def make_default_schedules(tubes, starts: Optional[Dict[int, float]] = None,
                           paragraph_seconds: float = 2.0) -> Dict[int, Schedule]:
    """Build trivial Schedules (speed=1, size=1) for a tube dict.

    Used when an optimiser only cares about start times — the renderer can
    still consume ``Dict[int, Schedule]`` and the result is identical to a
    plain start-time render.
    """
    starts = starts or {}
    return {
        tid: make_default_schedule(
            tube_id=tid,
            tube_duration=float(tube.duration),
            start=float(starts.get(tid, 0.0)),
            paragraph_seconds=paragraph_seconds,
        )
        for tid, tube in tubes.items()
        if getattr(tube, "num_frames", 0) > 0
    }

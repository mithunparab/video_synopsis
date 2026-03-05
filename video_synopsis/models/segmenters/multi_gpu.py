"""Multi-GPU segmenter wrapper — distributes frames across GPU instances."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List

import numpy as np

from video_synopsis.models.base import BaseSegmenter, SegmentationResult

log = logging.getLogger(__name__)


class MultiGPUSegmenter(BaseSegmenter):
    """Distributes frame batches across N single-GPU segmenter instances.

    Each GPU gets its own segmenter instance. Frames are split into contiguous
    chunks and processed in parallel via a thread pool (GPU ops release the GIL).
    """

    def __init__(
        self,
        factory_fn: Callable[[str], BaseSegmenter],
        gpu_ids: List[int],
    ):
        self._gpu_ids = gpu_ids
        # Initialize models sequentially to avoid race conditions
        # (e.g. concurrent weight downloads). Inference runs in parallel.
        self._segmenters = []
        for gid in gpu_ids:
            seg = factory_fn(f"cuda:{gid}")
            # Trigger model download/load now, while still sequential
            seg.segment_batch([np.zeros((64, 64, 3), dtype=np.uint8)])
            self._segmenters.append(seg)
            log.info(f"MultiGPUSegmenter: initialized GPU {gid}")
        self._pool = ThreadPoolExecutor(max_workers=len(gpu_ids))
        log.info(f"MultiGPUSegmenter: ready with {len(gpu_ids)} GPUs: {gpu_ids}")

    @property
    def num_gpus(self) -> int:
        return len(self._gpu_ids)

    def segment_batch(self, frames: List[np.ndarray]) -> List[SegmentationResult]:
        n = len(self._segmenters)
        if not frames:
            return []

        # Split frames into N contiguous chunks
        chunk_size = (len(frames) + n - 1) // n
        chunks = []
        chunk_indices = []  # track original ordering
        for i in range(n):
            start = i * chunk_size
            end = min(start + chunk_size, len(frames))
            if start < end:
                chunks.append(frames[start:end])
                chunk_indices.append((start, end))

        # Submit each chunk to its GPU's segmenter
        futures = [
            self._pool.submit(seg.segment_batch, chunk)
            for seg, chunk in zip(self._segmenters, chunks)
        ]

        # Gather results in order
        results: List[SegmentationResult] = []
        for future in futures:
            results.extend(future.result())

        return results

    def __del__(self):
        self._pool.shutdown(wait=False)

"""FastSAM segmenter via ultralytics."""

import logging
from typing import List

import cv2
import numpy as np

from video_synopsis.models.base import BaseSegmenter, SegmentationResult

log = logging.getLogger(__name__)


class FastSAMSegmenter(BaseSegmenter):
    """Instance segmentation using FastSAM (ultralytics).

    Requires: pip install ultralytics
    """

    def __init__(
        self,
        model_path: str = "FastSAM-s.pt",
        conf: float = 0.4,
        iou: float = 0.9,
        device: str = "",
        min_area: int = 500,
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.min_area = min_area
        self._device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import FastSAM
        except ImportError:
            raise ImportError(
                "FastSAM requires ultralytics. Install with: pip install ultralytics"
            )
        self._model = FastSAM(self.model_path)
        log.info(f"Loaded FastSAM model from {self.model_path}")

    def segment_batch(self, frames: List[np.ndarray]) -> List[SegmentationResult]:
        self._ensure_model()
        results = []

        for frame in frames:
            raw = self._model(
                frame,
                device=self._device or None,
                retina_masks=True,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )

            instance_masks = []
            instance_bboxes = []

            if raw and raw[0].masks is not None:
                masks_data = raw[0].masks.data.cpu().numpy()
                boxes_data = raw[0].boxes.xyxy.cpu().numpy()

                for i in range(len(boxes_data)):
                    x1, y1, x2, y2 = boxes_data[i][:4].astype(int)
                    w = x2 - x1
                    h = y2 - y1
                    if w * h < self.min_area:
                        continue

                    full_mask = (masks_data[i] > 0.5).astype(np.uint8) * 255
                    if full_mask.shape[:2] != frame.shape[:2]:
                        full_mask = cv2.resize(full_mask, (frame.shape[1], frame.shape[0]))

                    crop_mask = full_mask[y1:y2, x1:x2]
                    instance_masks.append(crop_mask)
                    instance_bboxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))

            results.append(SegmentationResult(
                masks=instance_masks,
                bboxes=instance_bboxes,
            ))

        return results

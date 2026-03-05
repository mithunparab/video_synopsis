"""RF-DETR segmenter — real-time DETR with instance segmentation."""

import logging
from typing import List

import cv2
import numpy as np

from video_synopsis.models.base import BaseSegmenter, SegmentationResult

log = logging.getLogger(__name__)


class RFDETRSegmenter(BaseSegmenter):
    """Instance segmentation using RF-DETR-Seg.

    RF-DETR is a real-time DETR-based detector (44.3 mAP @ 170 FPS on T4)
    with built-in instance segmentation support.

    Requires: pip install rfdetr
    """

    # COCO person class id
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_variant: str = "base",
        threshold: float = 0.5,
        device: str = "",
        min_area: int = 500,
    ):
        self.model_variant = model_variant
        self.threshold = threshold
        self.min_area = min_area
        self._device = device
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except ImportError:
            raise ImportError(
                "RF-DETR requires the rfdetr package. "
                "Install with: pip install rfdetr"
            )

        if self.model_variant == "large":
            self._model = RFDETRLarge()
        else:
            self._model = RFDETRBase()

        log.info(f"Loaded RF-DETR model (variant={self.model_variant})")

    def segment_batch(self, frames: List[np.ndarray]) -> List[SegmentationResult]:
        self._ensure_model()
        results = []

        for frame in frames:
            detections = self._model.predict(frame, threshold=self.threshold)

            instance_masks: List[np.ndarray] = []
            instance_bboxes: List[np.ndarray] = []
            instance_scores: List[float] = []

            boxes = detections.xyxy  # (N, 4)
            labels = detections.class_id  # (N,)
            scores = detections.confidence  # (N,)
            masks = getattr(detections, "mask", None)  # (N, H, W) or None

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    # Filter to person class only
                    if labels[i] != self.PERSON_CLASS_ID:
                        continue

                    x1, y1, x2, y2 = boxes[i][:4].astype(int)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    w = x2 - x1
                    h = y2 - y1
                    if w * h < self.min_area:
                        continue

                    if masks is not None and i < len(masks):
                        full_mask = (masks[i] > 0.5).astype(np.uint8) * 255
                        if full_mask.shape[:2] != frame.shape[:2]:
                            full_mask = cv2.resize(
                                full_mask,
                                (frame.shape[1], frame.shape[0]),
                            )
                        crop_mask = full_mask[y1:y2, x1:x2]
                    else:
                        # No mask available — use solid bbox mask
                        crop_mask = np.full((h, w), 255, dtype=np.uint8)

                    instance_masks.append(crop_mask)
                    instance_bboxes.append(
                        np.array([x1, y1, x2, y2], dtype=np.float32)
                    )
                    instance_scores.append(float(scores[i]))

            results.append(SegmentationResult(
                masks=instance_masks,
                bboxes=instance_bboxes,
                scores=instance_scores,
            ))

        return results

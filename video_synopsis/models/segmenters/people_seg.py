"""Wrapper around the people_segmentation model."""

import logging
from typing import List

import cv2
import numpy as np
import torch
from torch.amp import autocast

from video_synopsis.models.base import BaseSegmenter, SegmentationResult

log = logging.getLogger(__name__)


class PeopleSegmenter(BaseSegmenter):
    """People segmentation using pre-trained Unet model.

    Wraps the `people_segmentation` package with contour extraction
    to produce per-instance masks and bboxes.
    """

    def __init__(
        self,
        model_name: str = "Unet_2020-07-20",
        batch_size: int = 8,
        min_contour_fraction: float = 1 / 90,
        device: str = "",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.min_contour_fraction = min_contour_fraction

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from people_segmentation.pre_trained_models import create_model
        import albumentations as albu

        self._model = create_model(self.model_name).to(self.device)
        self._model.eval()
        self._transform = albu.Compose([albu.Normalize(p=1)], p=1)

    def _pad_image(self, image: np.ndarray):
        from iglovikov_helper_functions.utils.image_utils import pad
        return pad(image, factor=32, border=cv2.BORDER_CONSTANT)

    def segment_batch(self, frames: List[np.ndarray]) -> List[SegmentationResult]:
        self._ensure_model()
        from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

        results = []

        for start in range(0, len(frames), self.batch_size):
            batch_frames = frames[start : start + self.batch_size]
            tensors = []
            for frame in batch_frames:
                transformed = self._transform(image=frame)["image"]
                padded, _ = self._pad_image(transformed)
                tensors.append(tensor_from_rgb_image(padded))

            x = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                if self.device == "cuda":
                    with autocast(self.device):
                        preds = self._model(x)
                else:
                    preds = self._model(x)

            for i, pred in enumerate(preds):
                mask_full = (pred[0].cpu().numpy() > 0).astype(np.uint8)
                frame = batch_frames[i]
                h, w = frame.shape[:2]

                if mask_full.shape != (h, w):
                    mask_full = cv2.resize(mask_full, (w, h))

                # Extract contours as instances
                contours, _ = cv2.findContours(mask_full, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                min_area = (w * h) * self.min_contour_fraction

                instance_masks = []
                instance_bboxes = []
                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue
                    x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                    bbox = np.array([x_c, y_c, x_c + w_c, y_c + h_c], dtype=np.float32)

                    # Create per-instance mask
                    inst_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(inst_mask, [contour], -1, 255, -1)
                    # Crop to bbox region
                    inst_mask_crop = inst_mask[y_c : y_c + h_c, x_c : x_c + w_c]

                    instance_masks.append(inst_mask_crop)
                    instance_bboxes.append(bbox)

                results.append(SegmentationResult(
                    masks=instance_masks,
                    bboxes=instance_bboxes,
                ))

        return results

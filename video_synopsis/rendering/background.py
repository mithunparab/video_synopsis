"""Background extraction via median frame sampling."""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def extract_background(
    video_path: str,
    num_samples: int = 25,
    save_path: str = "",
) -> np.ndarray:
    """Extract background image by computing median of randomly sampled frames.

    Args:
        video_path: Path to input video.
        num_samples: Number of frames to sample.
        save_path: If provided, save the background image to this path.

    Returns:
        BGR background image as numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_samples = min(num_samples, total_frames)

    indices = np.random.choice(total_frames, size=num_samples, replace=False)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("No valid frames sampled for background extraction")

    # Pad frames to consistent size
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)

    padded = []
    for f in frames:
        h, w = f.shape[:2]
        top = (max_h - h) // 2
        bottom = max_h - h - top
        left = (max_w - w) // 2
        right = max_w - w - left
        padded.append(cv2.copyMakeBorder(f, top, bottom, left, right, cv2.BORDER_CONSTANT))

    background = np.median(np.array(padded), axis=0).astype(np.uint8)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, background)
        log.info(f"Background saved to {save_path}")

    return background

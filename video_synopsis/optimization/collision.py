"""Per-frame 3D collision detection for tube optimization.

The core fix: instead of computing IoU on static union bounding boxes
(which inflates a moving object to cover its entire path), we check
per-frame bboxes only during temporal overlap windows.
"""

import numpy as np

from video_synopsis.data.types import Tube


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1, y1, x2, y2] bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def _bbox_repulsion(a: np.ndarray, b: np.ndarray, sigma: float = 50.0) -> float:
    """Repulsion energy between two bboxes based on centroid distance."""
    cx_a = (a[0] + a[2]) / 2
    cy_a = (a[1] + a[3]) / 2
    cx_b = (b[0] + b[2]) / 2
    cy_b = (b[1] + b[3]) / 2
    dist_sq = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2
    return 1.0 / (dist_sq / (sigma ** 2 + 1e-6) + 1.0)


def compute_pairwise_collision_3d(
    tube_i: Tube,
    start_i: float,
    tube_j: Tube,
    start_j: float,
    method: str = "repulsion",
    sigma: float = 50.0,
    sample_step: int = 1,
) -> float:
    """Compute collision energy between two tubes at their optimized start times.

    This is the corrected version that uses **per-frame bounding boxes**
    instead of static union boxes.

    Algorithm:
        1. Shift each tube's timestamps by its optimized start time.
        2. Find the temporal overlap window.
        3. For overlapping frames, find nearest-neighbor bbox in each tube.
        4. Sum the spatial metric (IoU or repulsion) across overlap frames.

    Args:
        tube_i: First tube.
        start_i: Optimized start time for tube_i.
        tube_j: Second tube.
        start_j: Optimized start time for tube_j.
        method: 'iou' or 'repulsion'.
        sigma: Sigma for repulsion energy.
        sample_step: Sample every N-th overlapping frame (for speed).

    Returns:
        Total collision energy for this pair.
    """
    if tube_i.num_frames == 0 or tube_j.num_frames == 0:
        return 0.0

    # Shifted timestamps
    ts_i = tube_i.timestamps_array
    ts_j = tube_j.timestamps_array
    min_i = ts_i.min()
    min_j = ts_j.min()
    shifted_i = (ts_i - min_i) + start_i
    shifted_j = (ts_j - min_j) + start_j

    # Find temporal overlap
    overlap_start = max(shifted_i.min(), shifted_j.min())
    overlap_end = min(shifted_i.max(), shifted_j.max())
    if overlap_end < overlap_start:
        return 0.0  # No temporal overlap

    # Bboxes arrays
    bboxes_i = tube_i.bboxes_array
    bboxes_j = tube_j.bboxes_array

    # Handle single-point overlap (e.g. single-frame tubes at same time)
    if overlap_end == overlap_start:
        idx_i = int(np.argmin(np.abs(shifted_i - overlap_start)))
        idx_j = int(np.argmin(np.abs(shifted_j - overlap_start)))
        spatial_fn = _bbox_iou if method == "iou" else lambda a, b: _bbox_repulsion(a, b, sigma)
        return spatial_fn(bboxes_i[idx_i], bboxes_j[idx_j])

    # Generate sample times within the overlap window
    # Use the finer time resolution between the two tubes
    dt_i = np.diff(shifted_i).mean() if len(shifted_i) > 1 else 1.0
    dt_j = np.diff(shifted_j).mean() if len(shifted_j) > 1 else 1.0
    dt = min(abs(dt_i), abs(dt_j))
    if dt < 1e-6:
        dt = 1.0

    num_samples = max(1, int((overlap_end - overlap_start) / dt))
    sample_times = np.linspace(overlap_start, overlap_end, num_samples)
    if sample_step > 1:
        sample_times = sample_times[::sample_step]

    spatial_fn = _bbox_iou if method == "iou" else lambda a, b: _bbox_repulsion(a, b, sigma)

    total_collision = 0.0
    for t in sample_times:
        # Nearest-neighbor bbox lookup
        idx_i = int(np.argmin(np.abs(shifted_i - t)))
        idx_j = int(np.argmin(np.abs(shifted_j - t)))
        total_collision += spatial_fn(bboxes_i[idx_i], bboxes_j[idx_j])

    return total_collision


def compute_total_collision_3d(
    tubes: dict,
    starts: dict,
    method: str = "repulsion",
    sigma: float = 50.0,
    sample_step: int = 1,
) -> float:
    """Sum pairwise 3D collision over all tube pairs.

    Args:
        tubes: Dict[int, Tube].
        starts: Dict[int, float] — optimized start times.
        method: 'iou' or 'repulsion'.
        sigma: Sigma for repulsion.
        sample_step: Sampling step for speed.

    Returns:
        Total collision energy across all pairs.
    """
    ids = list(tubes.keys())
    total = 0.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ti, tj = ids[i], ids[j]
            if ti in starts and tj in starts:
                total += compute_pairwise_collision_3d(
                    tubes[ti], starts[ti],
                    tubes[tj], starts[tj],
                    method=method,
                    sigma=sigma,
                    sample_step=sample_step,
                )
    return total


def compute_energy(
    tubes: dict,
    starts: dict,
    w_duration: float = 1.0,
    w_collision: float = 10.0,
    w_activity: float = 0.1,
    method: str = "repulsion",
    sigma: float = 50.0,
    video_length: float = 0.0,
    sample_step: int = 1,
) -> float:
    """Compute total energy for a tube placement.

    E_total = w_duration * E_duration + w_collision * E_collision_3d + w_activity * E_activity

    Args:
        tubes: Dict[int, Tube].
        starts: Dict[int, float] — start times.
        w_duration: Weight for synopsis duration penalty.
        w_collision: Weight for collision penalty.
        w_activity: Weight for out-of-range penalty.
        method: 'iou' or 'repulsion'.
        sigma: Sigma for repulsion.
        video_length: Total video duration for activity penalty.
        sample_step: Sampling step for collision.

    Returns:
        Total energy (lower is better).
    """
    # E_duration: minimize synopsis length
    max_end = 0.0
    for tid, tube in tubes.items():
        if tid in starts and tube.num_frames > 0:
            end = starts[tid] + tube.duration
            max_end = max(max_end, end)
    e_duration = max_end

    # E_collision_3d
    e_collision = compute_total_collision_3d(
        tubes, starts, method=method, sigma=sigma, sample_step=sample_step
    )

    # E_activity: penalty for tubes placed outside valid range
    e_activity = 0.0
    if video_length > 0:
        for tid, start in starts.items():
            if tid in tubes:
                tube = tubes[tid]
                end = start + tube.duration
                if start < 0:
                    e_activity += abs(start)
                if end > video_length:
                    e_activity += end - video_length

    return w_duration * e_duration + w_collision * e_collision + w_activity * e_activity

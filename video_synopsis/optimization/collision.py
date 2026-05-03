"""Per-frame 3D collision detection for tube optimization.

The core fix: instead of computing IoU on static union bounding boxes
(which inflates a moving object to cover its entire path), we check
per-frame bboxes only during temporal overlap windows.

Two backends:
  * Numpy ``compute_energy`` / ``compute_pairwise_collision_3d`` — pure CPU,
    used by MCTS where many small per-state evaluations dominate.
  * Torch ``TubeBatch`` + ``compute_energy_torch`` — pre-uploads tubes to
    a GPU/MPS device, evaluates all O(n^2) pairs in parallel and supports
    a leading batch dimension on ``starts`` (used by Energy gradient
    finite-differences and PSO swarm fitness).
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from video_synopsis.data.types import Tube

log = logging.getLogger(__name__)


def pick_device(prefer: Optional[str] = None) -> torch.device:
    """cuda → mps → cpu, with optional override (e.g. 'cpu')."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _bbox_centroid_dist(a: np.ndarray, b: np.ndarray, radius: float = 30.0) -> float:
    """Smooth hinge on centroid distance: 1 - dist/R for dist < R, else 0.

    Pairs further than ``radius`` pixels apart contribute zero — they can run
    in parallel for free, even if their bounding boxes graze. Inside ``radius``
    the cost is linear in distance, giving gradient methods a usable signal.
    """
    cx_a = (a[0] + a[2]) / 2
    cy_a = (a[1] + a[3]) / 2
    cx_b = (b[0] + b[2]) / 2
    cy_b = (b[1] + b[3]) / 2
    dist = float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))
    return max(0.0, 1.0 - dist / max(radius, 1e-6))


def _spatial_fn_np(method: str, sigma: float, radius: float):
    if method == "iou":
        return _bbox_iou
    if method == "centroid":
        return lambda a, b: _bbox_centroid_dist(a, b, radius)
    return lambda a, b: _bbox_repulsion(a, b, sigma)


def compute_pairwise_collision_3d(
    tube_i: Tube,
    start_i: float,
    tube_j: Tube,
    start_j: float,
    method: str = "repulsion",
    sigma: float = 50.0,
    radius: float = 30.0,
    sample_step: int = 1,
) -> float:
    """Compute collision energy between two tubes at their optimized start times.

    Algorithm:
        1. Shift each tube's timestamps by its optimized start time.
        2. Find the temporal overlap window.
        3. For overlapping frames, find nearest-neighbor bbox in each tube.
        4. Sum the spatial metric across overlap frames.

    ``method``:
        * ``iou`` — bounding-box IoU (binary-feeling penalty).
        * ``repulsion`` — smooth ``1/(1+d²/σ²)`` proximity tax everywhere.
        * ``centroid`` — hinge ``max(0, 1 − d/R)``: zero when farther than R,
          linear inside R. Lets spatially-disjoint tubes overlap in time for
          free, while still strongly penalising true overlap.
    """
    if tube_i.num_frames == 0 or tube_j.num_frames == 0:
        return 0.0

    ts_i = tube_i.timestamps_array
    ts_j = tube_j.timestamps_array
    min_i = ts_i.min()
    min_j = ts_j.min()
    shifted_i = (ts_i - min_i) + start_i
    shifted_j = (ts_j - min_j) + start_j

    overlap_start = max(shifted_i.min(), shifted_j.min())
    overlap_end = min(shifted_i.max(), shifted_j.max())
    if overlap_end < overlap_start:
        return 0.0

    bboxes_i = tube_i.bboxes_array
    bboxes_j = tube_j.bboxes_array
    spatial_fn = _spatial_fn_np(method, sigma, radius)

    if overlap_end == overlap_start:
        idx_i = int(np.argmin(np.abs(shifted_i - overlap_start)))
        idx_j = int(np.argmin(np.abs(shifted_j - overlap_start)))
        return spatial_fn(bboxes_i[idx_i], bboxes_j[idx_j])

    dt_i = np.diff(shifted_i).mean() if len(shifted_i) > 1 else 1.0
    dt_j = np.diff(shifted_j).mean() if len(shifted_j) > 1 else 1.0
    dt = min(abs(dt_i), abs(dt_j))
    if dt < 1e-6:
        dt = 1.0

    num_samples = max(1, int((overlap_end - overlap_start) / dt))
    sample_times = np.linspace(overlap_start, overlap_end, num_samples)
    if sample_step > 1:
        sample_times = sample_times[::sample_step]

    total_collision = 0.0
    for t in sample_times:
        idx_i = int(np.argmin(np.abs(shifted_i - t)))
        idx_j = int(np.argmin(np.abs(shifted_j - t)))
        total_collision += spatial_fn(bboxes_i[idx_i], bboxes_j[idx_j])

    return total_collision


def compute_total_collision_3d(
    tubes: dict,
    starts: dict,
    method: str = "repulsion",
    sigma: float = 50.0,
    radius: float = 30.0,
    sample_step: int = 1,
) -> float:
    """Sum pairwise 3D collision over all tube pairs."""
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
                    radius=radius,
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
    radius: float = 30.0,
    video_length: float = 0.0,
    sample_step: int = 1,
) -> float:
    """Compute total energy for a tube placement (numpy / CPU).

    E_total = w_duration * E_duration + w_collision * E_collision_3d + w_activity * E_activity
    """
    max_end = 0.0
    for tid, tube in tubes.items():
        if tid in starts and tube.num_frames > 0:
            end = starts[tid] + tube.duration
            max_end = max(max_end, end)
    e_duration = max_end

    e_collision = compute_total_collision_3d(
        tubes, starts, method=method, sigma=sigma, radius=radius, sample_step=sample_step
    )

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


# --------------------------------------------------------------------------- #
# Torch backend                                                               #
# --------------------------------------------------------------------------- #


class TubeBatch:
    """Pre-uploaded torch representation of all tubes.

    All tensors live on ``device``. Padded to the longest tube; ``lengths``
    holds the true frame counts so masking is exact.
    """

    def __init__(
        self,
        tubes: Dict[int, Tube],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        ids = sorted(tubes.keys())
        self.ids = ids
        self.id_to_idx = {tid: i for i, tid in enumerate(ids)}
        N = len(ids)
        T_max = max((t.num_frames for t in tubes.values()), default=0)

        bboxes = torch.zeros((N, max(T_max, 1), 4), dtype=dtype)
        rel_ts = torch.zeros((N, max(T_max, 1)), dtype=dtype)
        lengths = torch.zeros((N,), dtype=torch.long)
        durations = torch.zeros((N,), dtype=dtype)

        for i, tid in enumerate(ids):
            tube = tubes[tid]
            n = tube.num_frames
            if n == 0:
                continue
            bb = torch.from_numpy(tube.bboxes_array.astype(np.float32))
            ts = torch.from_numpy(tube.timestamps_array.astype(np.float32))
            ts = ts - ts.min()
            bboxes[i, :n] = bb.to(dtype)
            rel_ts[i, :n] = ts.to(dtype)
            lengths[i] = n
            durations[i] = float(tube.duration)

        # Tail-fill rel_ts with the last valid value so searchsorted still
        # treats it as monotone non-decreasing past the tube length.
        for i in range(N):
            n = int(lengths[i])
            if 0 < n < T_max:
                rel_ts[i, n:] = rel_ts[i, n - 1]
                bboxes[i, n:] = bboxes[i, n - 1]

        self.bboxes = bboxes.to(device)
        self.rel_ts = rel_ts.to(device)
        self.lengths = lengths.to(device)
        self.durations = durations.to(device)

        if N >= 2:
            pi, pj = torch.triu_indices(N, N, offset=1)
            self.pair_i = pi.to(device)
            self.pair_j = pj.to(device)
        else:
            self.pair_i = torch.empty(0, dtype=torch.long, device=device)
            self.pair_j = torch.empty(0, dtype=torch.long, device=device)

        self.device = device
        self.dtype = dtype
        self.N = N
        self.T_max = max(T_max, 1)


def _bbox_iou_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU on [..., 4] tensors, returns [...]."""
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter
    return torch.where(union > 1e-6, inter / union, torch.zeros_like(inter))


def _bbox_repulsion_batched(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    cx_a = (a[..., 0] + a[..., 2]) / 2
    cy_a = (a[..., 1] + a[..., 3]) / 2
    cx_b = (b[..., 0] + b[..., 2]) / 2
    cy_b = (b[..., 1] + b[..., 3]) / 2
    dist_sq = (cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2
    return 1.0 / (dist_sq / (sigma ** 2 + 1e-6) + 1.0)


def _bbox_centroid_dist_batched(a: torch.Tensor, b: torch.Tensor, radius: float) -> torch.Tensor:
    """Hinge on centroid distance: 1 - dist/R inside R, else 0. See ``_bbox_centroid_dist``."""
    cx_a = (a[..., 0] + a[..., 2]) / 2
    cy_a = (a[..., 1] + a[..., 3]) / 2
    cx_b = (b[..., 0] + b[..., 2]) / 2
    cy_b = (b[..., 1] + b[..., 3]) / 2
    dist = torch.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2 + 1e-9)
    return (1.0 - dist / max(float(radius), 1e-6)).clamp(min=0.0)


def _gather_nearest(
    rel_ts: torch.Tensor,         # [P, T_max] sorted ascending, padded with last value
    bboxes: torch.Tensor,         # [P, T_max, 4]
    last_idx: torch.Tensor,       # [P] long, index of last valid frame (clamped >= 0)
    target: torch.Tensor,         # [B, P, K]
) -> torch.Tensor:
    """For each (b, p, k), return bboxes[p, nearest_index_in_rel_ts(target), :]."""
    B, P, K = target.shape
    T_max = rel_ts.shape[-1]

    # Reshape to [P, B*K] so searchsorted/gather work without expanding the
    # P-major tensors along the batch dim (saves memory).
    target_pbk = target.permute(1, 0, 2).reshape(P, B * K).contiguous()

    idx_right = torch.searchsorted(rel_ts, target_pbk)               # [P, B*K]
    idx_right = idx_right.clamp(max=T_max - 1)
    idx_left = (idx_right - 1).clamp(min=0)

    last_pbk = last_idx.unsqueeze(-1).expand(-1, B * K)               # [P, B*K]
    idx_right = torch.minimum(idx_right, last_pbk)
    idx_left = torch.minimum(idx_left, last_pbk)

    t_left = rel_ts.gather(1, idx_left)                              # [P, B*K]
    t_right = rel_ts.gather(1, idx_right)
    pick_right = (target_pbk - t_left).abs() > (target_pbk - t_right).abs()
    nearest = torch.where(pick_right, idx_right, idx_left)            # [P, B*K]

    nearest_4 = nearest.unsqueeze(-1).expand(-1, -1, 4)              # [P, B*K, 4]
    bbox_at_t = bboxes.gather(1, nearest_4)                          # [P, B*K, 4]
    return bbox_at_t.view(P, B, K, 4).permute(1, 0, 2, 3)            # [B, P, K, 4]


def _collision_chunk(
    starts_b: torch.Tensor,   # [B, N]
    pair_i: torch.Tensor,     # [P] long
    pair_j: torch.Tensor,     # [P] long
    bboxes_a: torch.Tensor,   # [P, T_max, 4]
    bboxes_b: torch.Tensor,
    rel_ts_a: torch.Tensor,   # [P, T_max]
    rel_ts_b: torch.Tensor,
    last_idx_a: torch.Tensor, # [P] long
    last_idx_b: torch.Tensor,
    max_rel_a: torch.Tensor,  # [P]
    max_rel_b: torch.Tensor,
    valid_pair: torch.Tensor, # [P] bool
    method: str,
    sigma: float,
    radius: float,
    K: int,
) -> torch.Tensor:
    """Inner batched kernel; all P-major tensors are precomputed."""
    start_a = starts_b.index_select(1, pair_i)                       # [B, P]
    start_b_ = starts_b.index_select(1, pair_j)

    a_lo = start_a
    a_hi = start_a + max_rel_a.unsqueeze(0)
    b_lo = start_b_
    b_hi = start_b_ + max_rel_b.unsqueeze(0)

    overlap_lo = torch.maximum(a_lo, b_lo)
    overlap_hi = torch.minimum(a_hi, b_hi)
    has_overlap = (overlap_hi > overlap_lo) & valid_pair.unsqueeze(0)  # [B, P]

    u = torch.linspace(0, 1, K, device=starts_b.device, dtype=starts_b.dtype)
    sample_times = overlap_lo.unsqueeze(-1) + u * (overlap_hi - overlap_lo).unsqueeze(-1)

    target_a = sample_times - start_a.unsqueeze(-1)
    target_b = sample_times - start_b_.unsqueeze(-1)

    bbox_a_at_t = _gather_nearest(rel_ts_a, bboxes_a, last_idx_a, target_a)
    bbox_b_at_t = _gather_nearest(rel_ts_b, bboxes_b, last_idx_b, target_b)

    if method == "iou":
        e = _bbox_iou_batched(bbox_a_at_t, bbox_b_at_t)
    elif method == "centroid":
        e = _bbox_centroid_dist_batched(bbox_a_at_t, bbox_b_at_t, radius)
    else:
        e = _bbox_repulsion_batched(bbox_a_at_t, bbox_b_at_t, sigma)

    e_per_pair = e.sum(dim=-1)
    e_per_pair = torch.where(has_overlap, e_per_pair, torch.zeros_like(e_per_pair))
    return e_per_pair.sum(dim=-1)


def compute_collision_energy_torch(
    batch: TubeBatch,
    starts: torch.Tensor,
    method: str = "repulsion",
    sigma: float = 50.0,
    radius: float = 30.0,
    sample_count: int = 32,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Pairwise collision energy summed over all tube pairs.

    Args:
        starts: ``[N]`` for a single placement, or ``[B, N]`` for a batch.
        chunk_size: process the batch dim in chunks of this size to bound
            peak memory. Output is independent of chunk size.

    Returns:
        ``[]`` (scalar) or ``[B]`` matching the input rank.
    """
    if batch.N < 2 or batch.pair_i.numel() == 0:
        if starts.ndim == 1:
            return torch.zeros((), device=batch.device, dtype=batch.dtype)
        return torch.zeros((starts.shape[0],), device=batch.device, dtype=batch.dtype)

    squeeze_at_end = starts.ndim == 1
    starts_b = starts.unsqueeze(0) if squeeze_at_end else starts
    B = starts_b.shape[0]
    K = max(1, int(sample_count))
    chunk = max(1, int(chunk_size))

    # Hoist per-pair tensors out of the chunk loop.
    pair_i = batch.pair_i
    pair_j = batch.pair_j
    bboxes_a = batch.bboxes.index_select(0, pair_i)
    bboxes_b = batch.bboxes.index_select(0, pair_j)
    rel_ts_a = batch.rel_ts.index_select(0, pair_i)
    rel_ts_b = batch.rel_ts.index_select(0, pair_j)
    len_a = batch.lengths.index_select(0, pair_i)
    len_b = batch.lengths.index_select(0, pair_j)
    last_idx_a = (len_a - 1).clamp(min=0)
    last_idx_b = (len_b - 1).clamp(min=0)
    max_rel_a = rel_ts_a.gather(1, last_idx_a.unsqueeze(-1)).squeeze(-1)
    max_rel_b = rel_ts_b.gather(1, last_idx_b.unsqueeze(-1)).squeeze(-1)
    valid_pair = (len_a > 0) & (len_b > 0)

    if B <= chunk:
        out = _collision_chunk(
            starts_b, pair_i, pair_j,
            bboxes_a, bboxes_b, rel_ts_a, rel_ts_b,
            last_idx_a, last_idx_b, max_rel_a, max_rel_b, valid_pair,
            method, sigma, radius, K,
        )
    else:
        parts = []
        for s in range(0, B, chunk):
            parts.append(_collision_chunk(
                starts_b[s:s + chunk], pair_i, pair_j,
                bboxes_a, bboxes_b, rel_ts_a, rel_ts_b,
                last_idx_a, last_idx_b, max_rel_a, max_rel_b, valid_pair,
                method, sigma, radius, K,
            ))
        out = torch.cat(parts, dim=0)

    return out.squeeze(0) if squeeze_at_end else out


def compute_energy_torch(
    batch: TubeBatch,
    starts: torch.Tensor,
    w_duration: float = 1.0,
    w_collision: float = 10.0,
    w_activity: float = 0.1,
    method: str = "repulsion",
    sigma: float = 50.0,
    radius: float = 30.0,
    video_length: float = 0.0,
    sample_count: int = 32,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Total placement energy.

    Args:
        starts: ``[N]`` or ``[B, N]``.
        chunk_size: collision-kernel batch chunk for memory bounding.

    Returns:
        ``[]`` or ``[B]`` matching input rank.
    """
    squeeze_at_end = starts.ndim == 1
    starts_b = starts.unsqueeze(0) if squeeze_at_end else starts

    ends = starts_b + batch.durations.unsqueeze(0)               # [B, N]
    e_duration = ends.max(dim=-1).values                          # [B]

    e_collision = compute_collision_energy_torch(
        batch, starts_b, method=method, sigma=sigma, radius=radius,
        sample_count=sample_count, chunk_size=chunk_size,
    )

    if video_length > 0:
        below = (-starts_b).clamp(min=0).sum(dim=-1)
        above = (ends - video_length).clamp(min=0).sum(dim=-1)
        e_activity = below + above
    else:
        e_activity = torch.zeros_like(e_duration)

    total = w_duration * e_duration + w_collision * e_collision + w_activity * e_activity
    return total.squeeze(0) if squeeze_at_end else total

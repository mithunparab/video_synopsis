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


def auto_tune_chronology_M(tubes: dict) -> float:
    """Auto-tune the chronology cutoff distance.

    Follows Nie et al. (TIP 2019): sort tubes by source-time first-frame, take
    the median of the +2-stride pairwise gaps. Tubes whose source first-frames
    are closer than M get strict order preservation; pairs farther apart can
    swap freely.
    """
    src_starts = sorted(
        float(t.timestamps_array.min()) for t in tubes.values() if t.num_frames > 0
    )
    if len(src_starts) < 3:
        return 1.0
    diffs = sorted(
        abs(src_starts[i + 2] - src_starts[i]) for i in range(len(src_starts) - 2)
    )
    return max(diffs[len(diffs) // 2], 1e-6)


def compute_chronology_np(
    tubes: dict,
    starts: dict,
    M: float,
) -> float:
    """Sum of inversion penalties (CPU). See ``compute_chronology_torch`` for
    the formula. Returns the unweighted total — the caller multiplies by
    ``w_chronology``.
    """
    if M <= 0:
        return 0.0
    ids = [tid for tid in tubes if tid in starts and tubes[tid].num_frames > 0]
    if len(ids) < 2:
        return 0.0
    src = [float(tubes[tid].timestamps_array.min()) for tid in ids]
    syn = [float(starts[tid]) for tid in ids]
    M_safe = max(float(M), 1e-6)
    cost = 0.0
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i == j:
                continue
            src_diff = src[j] - src[i]
            if src_diff <= 0:
                continue  # only count pairs where j is later in source
            syn_diff = syn[j] - syn[i]
            if syn_diff < 0:  # inverted
                proximity = max(0.0, 1.0 - src_diff / M_safe)
                cost += proximity * (-syn_diff)
    return cost


def compute_energy(
    tubes: dict,
    starts: dict,
    w_duration: float = 1.0,
    w_collision: float = 10.0,
    w_activity: float = 0.1,
    w_chronology: float = 0.0,
    chronology_M: float = 0.0,
    method: str = "repulsion",
    sigma: float = 50.0,
    radius: float = 30.0,
    video_length: float = 0.0,
    sample_step: int = 1,
) -> float:
    """Compute total energy for a tube placement (numpy / CPU).

    E_total = w_duration·E_dur + w_collision·E_col + w_activity·E_act + w_chronology·E_chron
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

    e_chronology = 0.0
    if w_chronology > 0 and chronology_M > 0:
        e_chronology = compute_chronology_np(tubes, starts, chronology_M)

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

    return (
        w_duration * e_duration
        + w_collision * e_collision
        + w_activity * e_activity
        + w_chronology * e_chronology
    )


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
        paragraph_seconds: float = 2.0,
    ):
        """``paragraph_seconds`` controls how each tube is split into paragraphs.
        ``K_max`` will be ``ceil(longest_tube_dur / paragraph_seconds)``. With
        ``paragraph_seconds`` very large, every tube is a single paragraph
        (legacy behaviour: speeds[N,1]=1, sizes[N,1]=1)."""
        ids = sorted(tubes.keys())
        self.ids = ids
        self.id_to_idx = {tid: i for i, tid in enumerate(ids)}
        N = len(ids)
        T_max = max((t.num_frames for t in tubes.values()), default=0)
        self.paragraph_seconds = float(paragraph_seconds)

        # Per-tube paragraph counts and the maximum across the batch (for padding).
        para_counts = []
        for tid in ids:
            tube = tubes[tid]
            if tube.num_frames == 0:
                para_counts.append(1)
            else:
                para_counts.append(max(1, int(np.ceil(tube.duration / max(paragraph_seconds, 1e-6)))))
        K_max = max(para_counts) if para_counts else 1

        bboxes = torch.zeros((N, max(T_max, 1), 4), dtype=dtype)
        rel_ts = torch.zeros((N, max(T_max, 1)), dtype=dtype)
        lengths = torch.zeros((N,), dtype=torch.long)
        durations = torch.zeros((N,), dtype=dtype)
        src_starts = torch.zeros((N,), dtype=dtype)  # source-time first-frame per tube
        # Paragraph tables; padded with src_dur=0 past the real K so a fully
        # vectorised kernel can divide by speed without touching invalid cells.
        para_lengths = torch.zeros((N,), dtype=torch.long)
        src_para_starts = torch.zeros((N, K_max), dtype=dtype)
        src_para_durs = torch.zeros((N, K_max), dtype=dtype)

        for i, tid in enumerate(ids):
            tube = tubes[tid]
            n = tube.num_frames
            if n == 0:
                continue
            bb = torch.from_numpy(tube.bboxes_array.astype(np.float32))
            ts = torch.from_numpy(tube.timestamps_array.astype(np.float32))
            src_starts[i] = float(ts.min())
            ts = ts - ts.min()
            bboxes[i, :n] = bb.to(dtype)
            rel_ts[i, :n] = ts.to(dtype)
            lengths[i] = n
            durations[i] = float(tube.duration)

            K = para_counts[i]
            para_lengths[i] = K
            base_dur = float(tube.duration) / K
            for k in range(K):
                src_para_starts[i, k] = k * base_dur
                src_para_durs[i, k] = base_dur
            src_para_durs[i, K - 1] = float(tube.duration) - (K - 1) * base_dur

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
        self.src_starts = src_starts.to(device)
        self.para_lengths = para_lengths.to(device)
        self.src_para_starts = src_para_starts.to(device)
        self.src_para_durs = src_para_durs.to(device)
        self.K_max = K_max

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


def _bbox_centroid_dist_batched_R(a: torch.Tensor, b: torch.Tensor, radius_per_sample: torch.Tensor) -> torch.Tensor:
    """Centroid hinge with per-sample radius. ``radius_per_sample`` broadcasts to ``a[..., 0]``."""
    cx_a = (a[..., 0] + a[..., 2]) / 2
    cy_a = (a[..., 1] + a[..., 3]) / 2
    cx_b = (b[..., 0] + b[..., 2]) / 2
    cy_b = (b[..., 1] + b[..., 3]) / 2
    dist = torch.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2 + 1e-9)
    R = radius_per_sample.clamp(min=1e-6)
    return (1.0 - dist / R).clamp(min=0.0)


def _scale_bbox_around_centroid(bbox: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Shrink each ``[..., 4]`` bbox by ``scale`` toward its centre.

    ``scale`` broadcasts to ``bbox[..., 0]``. Returns a new tensor.
    """
    cx = (bbox[..., 0] + bbox[..., 2]) / 2
    cy = (bbox[..., 1] + bbox[..., 3]) / 2
    half_w = (bbox[..., 2] - bbox[..., 0]) / 2 * scale
    half_h = (bbox[..., 3] - bbox[..., 1]) / 2 * scale
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def _paragraph_time_map(
    syn_offset: torch.Tensor,            # [B, P, Ks]
    cum_syn_starts: torch.Tensor,        # [B, P, Kp] cumulative synopsis-time start of each paragraph
    src_para_starts: torch.Tensor,       # [P, Kp] source-time start per paragraph
    speeds: torch.Tensor,                # [B, P, Kp]
    sizes: torch.Tensor,                 # [B, P, Kp]
    last_para_idx: torch.Tensor,         # [P] long, last valid paragraph index
):
    """Map per-sample synopsis offsets to source-time offsets via the paragraph map.

    Returns ``(t_src, size_at_t)`` both shape ``[B, P, Ks]``.
    """
    B, P, Ks = syn_offset.shape
    Kp = src_para_starts.shape[-1]

    # searchsorted requires monotone-ascending rows; cum_syn_starts is by construction.
    cum_flat = cum_syn_starts.reshape(B * P, Kp).contiguous()
    off_flat = syn_offset.reshape(B * P, Ks).contiguous()
    idx_flat = (torch.searchsorted(cum_flat, off_flat, right=True) - 1).clamp(min=0)  # [B*P, Ks]

    last_BP = last_para_idx.unsqueeze(0).expand(B, P).reshape(B * P).unsqueeze(-1)     # [B*P, 1]
    idx_flat = torch.minimum(idx_flat, last_BP)
    idx = idx_flat.reshape(B, P, Ks)                                                   # [B, P, Ks]

    cum_at = cum_syn_starts.gather(2, idx)
    speed_at = speeds.gather(2, idx)
    size_at = sizes.gather(2, idx)
    src_at = src_para_starts.unsqueeze(0).expand(B, P, Kp).gather(2, idx)

    t_src = src_at + (syn_offset - cum_at) * speed_at
    return t_src, size_at


def _synopsis_duration_per_tube(
    src_para_durs: torch.Tensor,    # [N, Kp]
    speeds: torch.Tensor,           # [B, N, Kp] or [N, Kp]
    para_lengths: torch.Tensor,     # [N]
) -> torch.Tensor:
    """``sum(src_dur[k] / speed[k])`` over valid paragraphs only. Returns ``[B, N]`` or ``[N]``."""
    Kp = src_para_durs.shape[-1]
    arange = torch.arange(Kp, device=src_para_durs.device).unsqueeze(0)
    valid = (arange < para_lengths.unsqueeze(-1)).to(src_para_durs.dtype)              # [N, Kp]
    if speeds.ndim == 3:
        per_para = src_para_durs.unsqueeze(0) / speeds.clamp(min=1e-6) * valid.unsqueeze(0)
        return per_para.sum(dim=-1)
    else:
        return ((src_para_durs / speeds.clamp(min=1e-6)) * valid).sum(dim=-1)


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
    syn_dur_a: torch.Tensor,  # [B, P]  synopsis duration of tube_a in each pair
    syn_dur_b: torch.Tensor,  # [B, P]
    cum_syn_a: torch.Tensor,  # [B, P, Kp] cumulative synopsis start per paragraph
    cum_syn_b: torch.Tensor,
    src_para_a: torch.Tensor, # [P, Kp]
    src_para_b: torch.Tensor,
    speeds_a: torch.Tensor,   # [B, P, Kp]
    speeds_b: torch.Tensor,
    sizes_a: torch.Tensor,    # [B, P, Kp]
    sizes_b: torch.Tensor,
    last_para_a: torch.Tensor,# [P] long
    last_para_b: torch.Tensor,
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
    a_hi = start_a + syn_dur_a
    b_lo = start_b_
    b_hi = start_b_ + syn_dur_b

    overlap_lo = torch.maximum(a_lo, b_lo)
    overlap_hi = torch.minimum(a_hi, b_hi)
    has_overlap = (overlap_hi > overlap_lo) & valid_pair.unsqueeze(0)  # [B, P]

    u = torch.linspace(0, 1, K, device=starts_b.device, dtype=starts_b.dtype)
    sample_times = overlap_lo.unsqueeze(-1) + u * (overlap_hi - overlap_lo).unsqueeze(-1)

    syn_off_a = sample_times - start_a.unsqueeze(-1)
    syn_off_b = sample_times - start_b_.unsqueeze(-1)

    t_src_a, size_a = _paragraph_time_map(syn_off_a, cum_syn_a, src_para_a, speeds_a, sizes_a, last_para_a)
    t_src_b, size_b = _paragraph_time_map(syn_off_b, cum_syn_b, src_para_b, speeds_b, sizes_b, last_para_b)

    bbox_a_at_t = _gather_nearest(rel_ts_a, bboxes_a, last_idx_a, t_src_a)
    bbox_b_at_t = _gather_nearest(rel_ts_b, bboxes_b, last_idx_b, t_src_b)

    if method == "iou":
        # IoU sees size by literally shrinking the bbox toward its centroid.
        bbox_a_s = _scale_bbox_around_centroid(bbox_a_at_t, size_a.unsqueeze(-1))
        bbox_b_s = _scale_bbox_around_centroid(bbox_b_at_t, size_b.unsqueeze(-1))
        e = _bbox_iou_batched(bbox_a_s, bbox_b_s)
    elif method == "centroid":
        # Centroid metric: shrinking bboxes doesn't move centroids, so size
        # affects collision via an effective radius `R_eff = R · (s_i + s_j)/2`.
        # Both shrunken → tubes can be closer; both full → standard R.
        R_eff = radius * (size_a + size_b) / 2.0
        e = _bbox_centroid_dist_batched_R(bbox_a_at_t, bbox_b_at_t, R_eff)
    else:
        # repulsion is centroid-based; size doesn't apply directly — leave as-is.
        e = _bbox_repulsion_batched(bbox_a_at_t, bbox_b_at_t, sigma)

    e_per_pair = e.sum(dim=-1)
    e_per_pair = torch.where(has_overlap, e_per_pair, torch.zeros_like(e_per_pair))
    return e_per_pair.sum(dim=-1)


def _prep_speeds_sizes(
    batch: TubeBatch,
    B: int,
    speeds: Optional[torch.Tensor],
    sizes: Optional[torch.Tensor],
) -> tuple:
    """Materialise ``speeds[B,N,Kp]`` and ``sizes[B,N,Kp]`` with sane defaults
    (1.0 everywhere) when caller passes ``None``."""
    Kp = batch.K_max
    N = batch.N
    if speeds is None:
        sp = torch.ones((B, N, Kp), device=batch.device, dtype=batch.dtype)
    elif speeds.ndim == 2:
        sp = speeds.unsqueeze(0).expand(B, N, Kp)
    else:
        sp = speeds
    if sizes is None:
        sz = torch.ones((B, N, Kp), device=batch.device, dtype=batch.dtype)
    elif sizes.ndim == 2:
        sz = sizes.unsqueeze(0).expand(B, N, Kp)
    else:
        sz = sizes
    return sp, sz


def compute_collision_energy_torch(
    batch: TubeBatch,
    starts: torch.Tensor,
    method: str = "repulsion",
    sigma: float = 50.0,
    radius: float = 30.0,
    sample_count: int = 32,
    chunk_size: int = 32,
    speeds: Optional[torch.Tensor] = None,
    sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise collision energy summed over all tube pairs.

    Args:
        starts: ``[N]`` for a single placement, or ``[B, N]`` for a batch.
        speeds, sizes: ``[N, Kp]`` (broadcast across batch) or ``[B, N, Kp]``,
            or ``None`` for ``1.0`` everywhere (legacy behaviour).
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

    speeds_BNK, sizes_BNK = _prep_speeds_sizes(batch, B, speeds, sizes)

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
    valid_pair = (len_a > 0) & (len_b > 0)

    src_para_a = batch.src_para_starts.index_select(0, pair_i)        # [P, Kp]
    src_para_b = batch.src_para_starts.index_select(0, pair_j)
    src_para_durs_a = batch.src_para_durs.index_select(0, pair_i)
    src_para_durs_b = batch.src_para_durs.index_select(0, pair_j)
    last_para_a = (batch.para_lengths.index_select(0, pair_i) - 1).clamp(min=0)
    last_para_b = (batch.para_lengths.index_select(0, pair_j) - 1).clamp(min=0)

    speeds_a_full = speeds_BNK.index_select(1, pair_i)                # [B, P, Kp]
    speeds_b_full = speeds_BNK.index_select(1, pair_j)
    sizes_a_full = sizes_BNK.index_select(1, pair_i)
    sizes_b_full = sizes_BNK.index_select(1, pair_j)

    syn_para_durs_a = src_para_durs_a.unsqueeze(0) / speeds_a_full.clamp(min=1e-6)
    syn_para_durs_b = src_para_durs_b.unsqueeze(0) / speeds_b_full.clamp(min=1e-6)
    cum_syn_a = torch.cumsum(syn_para_durs_a, dim=-1) - syn_para_durs_a
    cum_syn_b = torch.cumsum(syn_para_durs_b, dim=-1) - syn_para_durs_b
    Kp = batch.K_max
    arange_kp = torch.arange(Kp, device=batch.device).unsqueeze(0)
    valid_a = (arange_kp < batch.para_lengths.index_select(0, pair_i).unsqueeze(-1)).to(batch.dtype)
    valid_b = (arange_kp < batch.para_lengths.index_select(0, pair_j).unsqueeze(-1)).to(batch.dtype)
    syn_dur_a = (syn_para_durs_a * valid_a.unsqueeze(0)).sum(dim=-1)   # [B, P]
    syn_dur_b = (syn_para_durs_b * valid_b.unsqueeze(0)).sum(dim=-1)

    def _run(b_lo: int, b_hi: int) -> torch.Tensor:
        return _collision_chunk(
            starts_b[b_lo:b_hi], pair_i, pair_j,
            bboxes_a, bboxes_b, rel_ts_a, rel_ts_b,
            last_idx_a, last_idx_b,
            syn_dur_a[b_lo:b_hi], syn_dur_b[b_lo:b_hi],
            cum_syn_a[b_lo:b_hi], cum_syn_b[b_lo:b_hi],
            src_para_a, src_para_b,
            speeds_a_full[b_lo:b_hi], speeds_b_full[b_lo:b_hi],
            sizes_a_full[b_lo:b_hi], sizes_b_full[b_lo:b_hi],
            last_para_a, last_para_b,
            valid_pair,
            method, sigma, radius, K,
        )

    if B <= chunk:
        out = _run(0, B)
    else:
        parts = [_run(s, min(s + chunk, B)) for s in range(0, B, chunk)]
        out = torch.cat(parts, dim=0)

    return out.squeeze(0) if squeeze_at_end else out


def compute_chronology_torch(
    src_starts: torch.Tensor,
    syn_starts: torch.Tensor,
    M: float,
) -> torch.Tensor:
    """Inversion penalty: synopsis order should follow source order for pairs
    that were close together in source.

    For each pair (i, j) where ``src[j] > src[i]``, charge cost when
    ``syn[j] < syn[i]`` (inversion). The penalty scales with how much j is
    earlier in synopsis (linear) and how close i, j are in source
    (``max(0, 1 − Δsrc / M)``). Pairs farther than M apart in source are
    free to swap.

    ``syn_starts`` may be ``[N]`` or ``[B, N]``. Returns ``[]`` or ``[B]``.
    Auto-tune M with :func:`auto_tune_chronology_M`.
    """
    if M <= 0:
        if syn_starts.ndim == 1:
            return torch.zeros((), device=syn_starts.device, dtype=syn_starts.dtype)
        return torch.zeros((syn_starts.shape[0],), device=syn_starts.device, dtype=syn_starts.dtype)

    squeeze = syn_starts.ndim == 1
    syn = syn_starts.unsqueeze(0) if squeeze else syn_starts          # [B, N]

    src_diff = src_starts.unsqueeze(-2) - src_starts.unsqueeze(-1)    # [N, N]; src[j]-src[i]
    syn_diff = syn.unsqueeze(-2) - syn.unsqueeze(-1)                  # [B, N, N]; syn[j]-syn[i]

    M_safe = max(float(M), 1e-6)
    src_mask = (src_diff > 0).to(syn.dtype)
    proximity = (1.0 - src_diff.clamp(min=0) / M_safe).clamp(min=0.0) * src_mask   # [N, N]
    inversion = (-syn_diff).clamp(min=0.0)                             # [B, N, N]

    cost = (proximity.unsqueeze(0) * inversion).sum(dim=(-2, -1))     # [B]
    return cost.squeeze(0) if squeeze else cost


def compute_energy_torch(
    batch: TubeBatch,
    starts: torch.Tensor,
    w_duration: float = 1.0,
    w_collision: float = 10.0,
    w_activity: float = 0.1,
    w_chronology: float = 0.0,
    chronology_M: float = 0.0,
    method: str = "repulsion",
    sigma: float = 50.0,
    radius: float = 30.0,
    video_length: float = 0.0,
    sample_count: int = 32,
    chunk_size: int = 32,
    speeds: Optional[torch.Tensor] = None,
    sizes: Optional[torch.Tensor] = None,
    w_speed_reg: float = 0.0,
    w_size_reg: float = 0.0,
    speed_reg_alpha: float = 2.0,
    size_reg_alpha: float = 2.0,
) -> torch.Tensor:
    """Total placement energy.

    Args:
        starts: ``[N]`` or ``[B, N]``.
        speeds, sizes: optional ``[N, Kp]`` or ``[B, N, Kp]`` (default = 1.0).
        chunk_size: collision-kernel batch chunk for memory bounding.
        w_chronology, chronology_M: see :func:`compute_chronology_torch`.
        w_speed_reg, w_size_reg: regularisation on speed/size deviation from 1
            (``exp(α·max(s, 1/s)) − 1``-style cost). Without this the optimiser
            always picks the cheapest speed/size — typically extreme values that
            look bad at render time. Only active when ``speeds``/``sizes`` are
            provided.

    Returns:
        ``[]`` or ``[B]`` matching input rank.
    """
    squeeze_at_end = starts.ndim == 1
    starts_b = starts.unsqueeze(0) if squeeze_at_end else starts
    B = starts_b.shape[0]

    speeds_BNK, sizes_BNK = _prep_speeds_sizes(batch, B, speeds, sizes)
    syn_dur_per_tube = _synopsis_duration_per_tube(batch.src_para_durs, speeds_BNK, batch.para_lengths)
    ends = starts_b + syn_dur_per_tube                            # [B, N]
    e_duration = ends.max(dim=-1).values                          # [B]

    e_collision = compute_collision_energy_torch(
        batch, starts_b, method=method, sigma=sigma, radius=radius,
        sample_count=sample_count, chunk_size=chunk_size,
        speeds=speeds_BNK, sizes=sizes_BNK,
    )

    if video_length > 0:
        below = (-starts_b).clamp(min=0).sum(dim=-1)
        above = (ends - video_length).clamp(min=0).sum(dim=-1)
        e_activity = below + above
    else:
        e_activity = torch.zeros_like(e_duration)

    if w_chronology > 0 and chronology_M > 0:
        e_chronology = compute_chronology_torch(batch.src_starts, starts_b, chronology_M)
    else:
        e_chronology = torch.zeros_like(e_duration)

    e_reg = torch.zeros_like(e_duration)
    if (w_speed_reg > 0 or w_size_reg > 0) and (speeds is not None or sizes is not None):
        Kp = batch.K_max
        arange_kp = torch.arange(Kp, device=batch.device).unsqueeze(0)
        valid_NK = (arange_kp < batch.para_lengths.unsqueeze(-1)).to(batch.dtype)   # [N, Kp]
        if w_speed_reg > 0:
            sp = speeds_BNK.clamp(min=1e-6)
            # exp(α·max(s, 1/s) − α) − 1: 0 at s=1, grows in both directions.
            ratio = torch.maximum(sp, 1.0 / sp)
            cost = torch.exp(speed_reg_alpha * (ratio - 1.0)) - 1.0
            e_reg = e_reg + w_speed_reg * (cost * valid_NK.unsqueeze(0)).sum(dim=(-2, -1))
        if w_size_reg > 0:
            sz = sizes_BNK.clamp(min=1e-6)
            cost = torch.exp(size_reg_alpha * (1.0 / sz - 1.0)) - 1.0
            e_reg = e_reg + w_size_reg * (cost * valid_NK.unsqueeze(0)).sum(dim=(-2, -1))

    total = (
        w_duration * e_duration
        + w_collision * e_collision
        + w_activity * e_activity
        + w_chronology * e_chronology
        + e_reg
    )
    return total.squeeze(0) if squeeze_at_end else total

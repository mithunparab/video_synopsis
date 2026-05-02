"""Shared 3D scatter plots for tube placements.

Each optimizer produces a `placements: Dict[tube_id -> start_time]`. We render
those placements (plus the original arrangement) as 3D X/Y/Time scatter plots
so different methods can be compared visually.
"""

import logging
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from video_synopsis.data.types import Tube

log = logging.getLogger(__name__)


def _scatter_arrangement(
    ax,
    tubes: Dict[int, Tube],
    placements: Optional[Dict[int, float]],
    title: str,
    show_legend: bool = True,
) -> None:
    """Draw one 3D X/Y/Time scatter into `ax`.

    If `placements` is None, plots tubes at their original timestamps.
    Otherwise shifts each tube so its first frame lands at placements[tid].
    """
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time")

    colors = plt.colormaps.get_cmap("tab20")
    n = max(len(tubes), 1)

    for i, (tid, tube) in enumerate(tubes.items()):
        if tube.num_frames == 0:
            continue
        bboxes = tube.bboxes_array
        ts = tube.timestamps_array
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        color = colors(i / n)

        if placements is None:
            zs = ts
        elif tid in placements:
            zs = (ts - ts.min()) + placements[tid]
        else:
            continue

        ax.scatter(cx, cy, zs, s=10, color=color, label=f"Tube {tid}")

    if show_legend and len(tubes) <= 20:
        ax.legend(fontsize="small")


def save_initial_vs_optimized(
    tubes: Dict[int, Tube],
    placements: Dict[int, float],
    output_path: str,
    method_name: str,
) -> None:
    """Save a 2-panel plot: original arrangement vs. optimized arrangement.

    The figure suptitle names the method so the saved PNG is self-describing.
    """
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"Tube placement — {method_name}", fontsize=16, fontweight="bold")

        ax1 = fig.add_subplot(121, projection="3d")
        _scatter_arrangement(ax1, tubes, None, "Initial Arrangement")

        ax2 = fig.add_subplot(122, projection="3d")
        _scatter_arrangement(ax2, tubes, placements, f"Optimized — {method_name}")

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        log.info(f"Plot saved to {output_path}")
    except Exception as e:
        log.warning(f"Failed to save plot: {e}")


def save_methods_comparison(
    tubes: Dict[int, Tube],
    placements_per_method: Dict[str, Dict[int, float]],
    output_path: str,
) -> None:
    """Save a single figure comparing the original arrangement against each method.

    One subplot per method, plus a leading "Initial" panel. Layout adapts to the
    number of methods (up to 3 columns).
    """
    if not placements_per_method:
        log.warning("No methods to compare — skipping comparison plot.")
        return

    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        method_names = list(placements_per_method.keys())
        n_panels = len(method_names) + 1  # +1 for initial
        ncols = min(3, n_panels)
        nrows = (n_panels + ncols - 1) // ncols

        fig = plt.figure(figsize=(8 * ncols, 7 * nrows))
        fig.suptitle("Tube placement — method comparison", fontsize=16, fontweight="bold")

        ax = fig.add_subplot(nrows, ncols, 1, projection="3d")
        _scatter_arrangement(ax, tubes, None, "Initial Arrangement", show_legend=False)

        for idx, name in enumerate(method_names, start=2):
            ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
            placements = placements_per_method[name]
            synopsis_len = _synopsis_length(tubes, placements)
            title = f"{name}  (synopsis: {synopsis_len:.1f}s)"
            _scatter_arrangement(ax, tubes, placements, title, show_legend=False)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        log.info(f"Comparison plot saved to {output_path}")
    except Exception as e:
        log.warning(f"Failed to save comparison plot: {e}")


def _synopsis_length(tubes: Dict[int, Tube], placements: Dict[int, float]) -> float:
    end = 0.0
    for tid, start in placements.items():
        tube = tubes.get(tid)
        if tube is None or tube.num_frames == 0:
            continue
        end = max(end, start + tube.duration)
    return end

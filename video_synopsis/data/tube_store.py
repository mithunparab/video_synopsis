import os
import glob
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from video_synopsis.data.types import Tube, TubeFrame

log = logging.getLogger(__name__)


class TubeArchive:
    """Efficient tube storage using .npz files instead of per-frame PNG/CSV."""

    @staticmethod
    def save(tube: Tube, path: str) -> None:
        """Save a single Tube to a .npz file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if tube.num_frames == 0:
            log.warning(f"Tube {tube.tube_id} has no frames, skipping save.")
            return

        bboxes = tube.bboxes_array
        timestamps = tube.timestamps_array
        frame_indices = tube.frame_indices_array

        # Pack images and masks into object arrays (variable size per frame)
        images = np.array([f.image for f in tube.frames], dtype=object)
        masks = np.array([f.mask for f in tube.frames], dtype=object)

        np.savez_compressed(
            path,
            tube_id=np.array([tube.tube_id]),
            bboxes=bboxes,
            timestamps=timestamps,
            frame_indices=frame_indices,
            images=images,
            masks=masks,
        )
        log.debug(f"Saved tube {tube.tube_id} ({tube.num_frames} frames) to {path}")

    @staticmethod
    def load(path: str) -> Tube:
        """Load a Tube from a .npz file."""
        data = np.load(path, allow_pickle=True)
        tube_id = int(data["tube_id"][0])
        bboxes = data["bboxes"]
        timestamps = data["timestamps"]
        frame_indices = data["frame_indices"]
        images = data["images"]
        masks = data["masks"]

        frames = []
        for i in range(len(bboxes)):
            frames.append(TubeFrame(
                frame_index=int(frame_indices[i]),
                bbox=bboxes[i],
                mask=masks[i],
                image=images[i],
                timestamp=float(timestamps[i]),
            ))

        tube = Tube(tube_id=tube_id, frames=frames)
        log.debug(f"Loaded tube {tube_id} ({len(frames)} frames) from {path}")
        return tube

    @staticmethod
    def save_all(tubes: Dict[int, Tube], directory: str) -> None:
        """Save all tubes into a directory."""
        os.makedirs(directory, exist_ok=True)
        for tube_id, tube in tubes.items():
            path = os.path.join(directory, f"tube_{tube_id:04d}.npz")
            TubeArchive.save(tube, path)

    @staticmethod
    def load_all(directory: str) -> Dict[int, Tube]:
        """Load all tubes from a directory."""
        tubes = {}
        for path in sorted(glob.glob(os.path.join(directory, "tube_*.npz"))):
            tube = TubeArchive.load(path)
            tubes[tube.tube_id] = tube
        return tubes

    @staticmethod
    def from_legacy_dirs(
        output_dir: str,
        mask_dir: str,
        ext: str = ".png",
    ) -> Dict[int, Tube]:
        """Load tubes from legacy per-frame PNG + CSV directory structure.

        Expected layout:
            output_dir/XXXX/XXXXnode.csv  (tube metadata)
            output_dir/XXXX/NNNN.png      (ROI images)
            mask_dir/XXXX/NNNN.png        (masks)
        """
        tubes: Dict[int, Tube] = {}
        tube_dirs = sorted(glob.glob(os.path.join(output_dir, "[0-9]*")))

        for tube_dir in tube_dirs:
            if not os.path.isdir(tube_dir):
                continue
            tube_id_str = os.path.basename(tube_dir)
            try:
                tube_id = int(tube_id_str)
            except ValueError:
                continue

            csv_files = glob.glob(os.path.join(tube_dir, "*node.csv"))
            if not csv_files:
                continue

            df = pd.read_csv(csv_files[0])
            tube = Tube(tube_id=tube_id)

            for _, row in df.iterrows():
                n = int(row["n"])
                x1, y1 = int(row["x1"]), int(row["y1"])
                x2, y2 = int(row["x2"]), int(row["y2"])
                timestamp = float(row["time"])

                img_path = os.path.join(tube_dir, f"{n:04d}{ext}")
                msk_path = os.path.join(mask_dir, tube_id_str, f"{n:04d}{ext}")

                if not os.path.exists(img_path) or not os.path.exists(msk_path):
                    continue

                image = cv2.imread(img_path)
                mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                if image is None or mask is None:
                    continue

                tube.add_frame(TubeFrame(
                    frame_index=n,
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    mask=mask,
                    image=image,
                    timestamp=timestamp,
                ))

            if tube.num_frames > 0:
                tubes[tube_id] = tube

        log.info(f"Loaded {len(tubes)} tubes from legacy directories")
        return tubes

    @staticmethod
    def to_legacy_csv(
        tubes: Dict[int, Tube],
        optimized_starts: Dict[int, float],
        output_dir: str,
    ) -> None:
        """Write optimized tube data as CSVs compatible with the legacy stitcher."""
        os.makedirs(output_dir, exist_ok=True)
        for tube_id, tube in tubes.items():
            if tube.num_frames == 0:
                continue
            start_offset = optimized_starts.get(tube_id, 0.0)
            min_ts = tube.start_time

            rows = []
            for f in tube.frames:
                new_time = (f.timestamp - min_ts) + start_offset
                rows.append({
                    "T": tube_id,
                    "n": f.frame_index,
                    "x1": int(f.bbox[0]),
                    "y1": int(f.bbox[1]),
                    "x2": int(f.bbox[2]),
                    "y2": int(f.bbox[3]),
                    "time": new_time,
                })

            df = pd.DataFrame(rows)
            csv_path = os.path.join(output_dir, f"optimized_tube_{tube_id}.csv")
            df.to_csv(csv_path, index=False)

    @staticmethod
    def to_legacy_txt(
        tubes: Dict[int, Tube],
        optimized_starts: Dict[int, float],
        output_dir: str,
    ) -> None:
        """Write optimized tubes as TXT files for the legacy stitcher."""
        os.makedirs(output_dir, exist_ok=True)
        for tube_id, tube in tubes.items():
            if tube.num_frames == 0:
                continue
            start_offset = optimized_starts.get(tube_id, 0.0)
            min_ts = tube.start_time

            lines = []
            for f in tube.frames:
                new_time = (f.timestamp - min_ts) + start_offset
                lines.append(
                    f"{tube_id}, {f.frame_index}, "
                    f"{int(f.bbox[0])}, {int(f.bbox[2])}, "
                    f"{int(f.bbox[1])}, {int(f.bbox[3])}, "
                    f"{new_time:.2f},"
                )

            txt_path = os.path.join(output_dir, f"optimized_tube_{tube_id}.txt")
            with open(txt_path, "w") as fout:
                fout.write("\n".join(lines) + "\n")

# Video Synopsis



---

*Flow of video synopsis framework [1]*

---

A framework that condenses long surveillance/security videos into short, non-chronological summaries. It detects people, tracks them as "tubes" (spatial-temporal trajectories), optimizes tube placement to minimize temporal overlap, and composites them onto a background image.

## Installation

```bash
pip install -r requirements.txt
```

Optional dependencies:

```bash
# RF-DETR segmenter (default)
pip install rfdetr

# BoT-SORT tracker (default)
pip install boxmot

# FastSAM segmenter
pip install ultralytics

# SAM3 tracker
pip install transformers>=4.40

# All optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Python API

```python
from video_synopsis import Pipeline, Config

config = Config(video="path/to/video.mp4", optimizer="mcts", epochs=2000)
pipeline = Pipeline(config)
output_path = pipeline.run()
```

### CLI

```bash
# Using the package module
python -m video_synopsis.cli -v path/to/video.mp4

# Using the legacy entry point
python main.py -v path/to/video.mp4 --epochs 2000
```

### WebUI (Streamlit)

```bash
streamlit run app.py
```

## Architecture

### Pipeline Flow

```
Input Video -> Background Extraction (median frame)
            -> Batch Inference (RF-DETR / people-seg / FastSAM)
            -> Tracking (BoT-SORT / SORT / SAM3)
            -> Tube Generation (per-object ROI + mask)
            -> Optimization (MCTS+AlphaZero, Energy-based, or PSO)
            -> Tube Stitching (composite onto background)
            -> Output MP4
```

### Package Structure

```
video_synopsis/
├── __init__.py              # Exposes Pipeline, Config
├── pipeline.py              # Main orchestrator
├── config.py                # Dataclass-based configuration
├── data/
│   ├── types.py             # TubeFrame, Tube dataclasses
│   └── tube_store.py        # TubeArchive: .npz read/write
├── models/
│   ├── base.py              # BaseSegmenter, BaseTracker ABCs
│   ├── segmenters/
│   │   ├── rfdetr.py        # RF-DETR segmenter (default)
│   │   ├── people_seg.py    # People segmentation (Unet)
│   │   └── fastsam.py       # FastSAM (ultralytics)
│   └── trackers/
│       ├── botsort_tracker.py # BoT-SORT with ReID (default)
│       ├── sort_tracker.py  # SORT (Kalman + Hungarian)
│       └── sam3_tracker.py  # SAM3 (transformers)
├── optimization/
│   ├── base.py              # BaseOptimizer ABC
│   ├── collision.py         # Per-frame 3D collision detection
│   ├── energy.py            # Gradient-based optimizer
│   ├── mcts.py              # MCTS+AlphaZero optimizer
│   └── pso.py               # Particle Swarm Optimization
├── rendering/
│   ├── background.py        # Median-frame background extraction
│   └── stitcher.py          # Tube compositing
├── cli.py                   # CLI entry point
└── webui.py                 # Streamlit WebUI
```

### Pluggable Models

**Segmenters** (`--segmenter`):

- `rfdetr` (default) - RF-DETR real-time DETR segmenter, 44.3 mAP @ 170 FPS (requires `pip install rfdetr`)
- `people` - Pre-trained people segmentation Unet
- `fastsam` - FastSAM via ultralytics (requires `pip install ultralytics`)

**Trackers** (`--tracker`):

- `botsort` (default) - BoT-SORT with ReID, MOT leaderboard #1 (requires `pip install boxmot`)
- `sort` - SORT with Kalman filter
- `sam3` - SAM3 tracker via transformers (requires `pip install transformers>=4.40`)

**Optimizers** (`--optimizer`):

- `mcts` (default) - MCTS+AlphaZero with neural network guidance
- `energy` - Gradient-based energy minimization
- `pso` - Particle Swarm Optimization (population-based, no gradient or NN overhead)

### Data Storage

Tubes are stored as `.npz` archives (bboxes, timestamps, images, masks in a single file) instead of per-frame PNG + CSV files. This reduces I/O overhead significantly. Legacy format is still supported via `TubeArchive.from_legacy_dirs()`.

### 3D Collision Detection

The optimizer uses **per-frame bounding boxes** for collision detection. For each pair of tubes, it:

1. Shifts timestamps by optimized start times
2. Finds the temporal overlap window
3. Computes spatial overlap (IoU or repulsion) at each overlapping frame
4. Sums the collision energy

This avoids the false-positive problem where a static union bbox inflates a moving object's footprint across its entire path.

## CLI Options

```
python -m video_synopsis.cli -h

-v, --video              Input video path
-f, --fps                Frames per second (default: 25)
-bsz, --batch_size       Inference batch size (default: 8)
--segmenter              Segmentation model: rfdetr, people, fastsam
--tracker                Tracker: botsort, sort, sam3
--rfdetr_variant         RF-DETR model: base, large (default: base)
--rfdetr_threshold       RF-DETR confidence threshold (default: 0.5)
--botsort_track_high_thresh  BoT-SORT high detection threshold (default: 0.6)
--botsort_track_buffer   BoT-SORT track buffer frames (default: 30)
--botsort_match_thresh   BoT-SORT matching threshold (default: 0.8)
--botsort_no_reid        Disable BoT-SORT ReID features
--optimizer              Optimization method: mcts, energy, pso
--epochs                 Optimization epochs (default: 2000)
--collision_method       Collision metric: repulsion, iou
--sigma                  Repulsion sigma (default: 50.0)
--pso_num_particles      PSO swarm size (default: 30)
--pso_max_iterations     PSO iteration limit (default: 500)
--pso_inertia            PSO inertia weight (default: 0.7)
--pso_cognitive          PSO cognitive coefficient (default: 1.5)
--pso_social             PSO social coefficient (default: 1.5)
--energy_optimization    Enable optimization (default: True)
--use_npz                Use .npz storage (default: True)
```

## Result

Sample video demonstrates the framework's effectiveness:

[Download Video](https://drive.google.com/file/d/1ZqZ9bVY75VbuRh_A1Qfzaw0iLKpZV6X9/view?usp=sharing)


| Original Video Time (sec) | Video Synopsis Time (sec) |
| ------------------------- | ------------------------- |
| 49                        | 6                         |


Initial arrangement of tubes (49 sec) vs Optimized arrangement (6 sec)

## Dependencies

- PyTorch (inference + MCTS optimization)
- OpenCV (video I/O, requires H.264/libx264 codec)
- `people-segmentation` with model `Unet_2020-07-20` (auto-downloads on first run)
- `filterpy` for Kalman filtering in SORT tracker

## References

[1] Nonchronological Video Synopsis and Indexing - Yael Pritch, Alex Rav-Acha, and Shmuel Peleg, IEEE
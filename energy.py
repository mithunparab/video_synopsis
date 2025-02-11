import logging
import os
import re
import shutil
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import cv2
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger('tube_optimizer')
log.setLevel(logging.INFO)

def plot_tubes(ax: plt.Axes, tube_data: Dict[int, np.ndarray], tube_starts: Optional[List[float]], 
               areas: Dict[int, float], title: str) -> None:
    colors = plt.cm.viridis(np.linspace(0, 1, len(tube_data)))
    for i, (tube_id, points) in enumerate(tube_data.items()):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        if tube_starts is not None:
            z = z - np.min(z) + tube_starts[i]
        area_size = (areas[tube_id] / np.max(list(areas.values())) * 100)
        ax.scatter(x, y, z, s=area_size, c=[colors[i % len(colors)]], label=f'Tube {tube_id}')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Time (t)')

def parse_contours(contour_str: str) -> np.ndarray:
    points = re.findall(r'\[\[\[(\d+)\s+(\d+)\]\]\]', contour_str)
    array_points = [tuple(map(int, point)) for point in points]
    return np.array(array_points, dtype=np.int32).reshape(-1, 1, 2)

def compute_convex_hull(contours: np.ndarray) -> np.ndarray:
    return cv2.convexHull(contours) if contours.shape[0] >= 3 else contours

def load_tubes(files_pattern: str) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, Tuple[float, ...]], Dict[int, int], Dict[int, pd.DataFrame]]:
    tube_data = {}
    areas = {}
    boxes = {}
    original_tube_map = {}
    full_dataframes = {}
    tube_id_counter = 0
    
    for file_path in glob.glob(files_pattern):
        file_name = os.path.basename(file_path)
        original_tube_id = int(file_name.split('node')[0]) 
        df = pd.read_csv(file_path)
        
        # Data processing
        df['centroid_x'] = (df['x1'] + df['x2']) / 2
        df['centroid_y'] = (df['y1'] + df['y2']) / 2
        df['area'] = abs((df['x2'] - df['x1']) * (df['y2'] - df['y1']))
        df_sorted = df.sort_values(by='time')
        
        # Split tubes at time gaps
        time_diffs = df_sorted['time'].diff()
        split_points = time_diffs > 1

        for is_new_tube, group in df_sorted.groupby(split_points.cumsum()):
            tube_values = group[['centroid_x', 'centroid_y', 'time']].values
            tube_data[tube_id_counter] = tube_values
            areas[tube_id_counter] = group['area'].mean()
            
            # Bounding box calculation
            min_cx, min_cy = group['centroid_x'].min(), group['centroid_y'].min()
            max_cx, max_cy = group['centroid_x'].max(), group['centroid_y'].max()
            boxes[tube_id_counter] = (min_cx, min_cy, max_cx, max_cy)
            
            # Metadata storage
            original_tube_map[tube_id_counter] = original_tube_id
            full_dataframes[tube_id_counter] = group
            tube_id_counter += 1

    return tube_data, areas, boxes, original_tube_map, full_dataframes

def compute_vectorized_energy(starts: torch.Tensor, durations: torch.Tensor, 
                             boxes_tensor: torch.Tensor, total_frames: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Vectorized energy computation using PyTorch tensor operations."""
    n_tubes = starts.shape[0]
    ends = starts + durations

    # Collision energy calculation
    start_i = starts.unsqueeze(1)
    start_j = starts.unsqueeze(0)
    end_i = ends.unsqueeze(1)
    end_j = ends.unsqueeze(0)

    # Temporal overlap matrix
    overlap_start = torch.maximum(start_i, start_j)
    overlap_end = torch.minimum(end_i, end_j)
    overlap_time = torch.clamp(overlap_end - overlap_start, min=0)

    # Bounding box IoU matrix
    box_i = boxes_tensor.unsqueeze(1)
    box_j = boxes_tensor.unsqueeze(0)
    
    xA = torch.max(box_i[..., 0], box_j[..., 0])
    yA = torch.max(box_i[..., 1], box_j[..., 1])
    xB = torch.min(box_i[..., 2], box_j[..., 2])
    yB = torch.min(box_i[..., 3], box_j[..., 3])

    inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    box_areas = (boxes_tensor[..., 2] - boxes_tensor[..., 0]) * (boxes_tensor[..., 3] - boxes_tensor[..., 1])
    union_area = box_areas.unsqueeze(1) + box_areas.unsqueeze(0) - inter_area
    iou = torch.zeros_like(inter_area, device=device)
    valid = union_area > 0
    iou[valid] = inter_area[valid] / union_area[valid]

    # Energy components
    mask = ~torch.eye(n_tubes, dtype=torch.bool, device=device)
    collision_energy = (iou * overlap_time)[mask].sum() / 2  # Account for double-counting

    # Chronological energy
    contained = ((start_i > start_j) & (end_i < end_j)) | ((start_i < start_j) & (end_i > end_j))
    chronological_energy = (torch.abs(start_i - start_j) * contained.float())[mask].sum() / 2

    # Activity energy
    video_end = torch.tensor(total_frames, dtype=torch.float32, device=device)
    activity_energy = (
        torch.clamp(-starts, min=0).sum() +  # Starts before video beginning
        torch.clamp(ends - video_end, min=0).sum()  # Ends after video end
    )

    # Temporal energy
    sorted_indices = torch.argsort(starts)
    sorted_ends = ends[sorted_indices]
    temporal_gaps = starts[sorted_indices][1:] - sorted_ends[:-1]
    temporal_energy = torch.clamp(temporal_gaps, min=0).pow(2).sum()

    total_energy = collision_energy + chronological_energy + activity_energy + temporal_energy
    return total_energy, collision_energy, chronological_energy, activity_energy, temporal_energy

def optimize_tube_starts(tube_data: Dict[int, np.ndarray], areas: Dict[int, float], 
                        boxes: Dict[int, Tuple[float, ...]], total_frames: int = 1474, 
                        epochs: int = 1000, device: str = 'cpu') -> Dict[int, float]:
    # Device setup
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    tube_ids = list(tube_data.keys())
    if not tube_ids:
        log.warning("No tubes to optimize")
        return {}

    # Precompute tensors on target device
    durations = torch.tensor(
        [tube[..., 2].ptp() for tube in tube_data.values()],
        dtype=torch.float32,
        device=device
    )
    boxes_tensor = torch.tensor(
        [boxes[tid] for tid in tube_ids],
        dtype=torch.float32,
        device=device
    )

    # Optimization setup
    starts = torch.nn.Parameter(torch.zeros(len(tube_ids), device=device))
    optimizer = torch.optim.Adam([starts], lr=0.01)
    best_energy = float('inf')
    best_starts = None

    for epoch in tqdm(range(epochs), desc='Optimization'):
        optimizer.zero_grad()
        
        # Sort starts for temporal consistency
        sorted_starts, _ = torch.sort(starts)
        total_energy, *components = compute_vectorized_energy(
            sorted_starts, durations, boxes_tensor, total_frames, device
        )
        
        total_energy.backward()
        optimizer.step()
        with torch.no_grad():
            starts.clamp_(min=0)
            
            # Track best solution
            if total_energy < best_energy:
                best_energy = total_energy.item()
                best_starts = sorted_starts.detach().clone()

        if epoch % 100 == 0:
            log.info(f"Epoch {epoch}: Energy={total_energy.item():.2f} "
                     f"(Collision={components[0].item():.2f}, "
                     f"Temporal={components[-1].item():.2f})")

    # Return best found solution
    final_starts = best_starts.cpu().numpy()
    return {tid: final_starts[i] for i, tid in enumerate(tube_ids)}

def save_optimized_tubes(full_dataframes: Dict[int, pd.DataFrame], 
                        optimized_starts: Dict[int, float], 
                        original_tube_map: Dict[int, int], 
                        output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for tube_id, df in full_dataframes.items():
        original_id = original_tube_map[tube_id]
        start_time = optimized_starts[tube_id]
        time_offset = start_time - df['time'].min()
        
        df = df.copy()
        df['time'] += time_offset
        output_path = os.path.join(output_dir, f"optimized_tube_{original_id}.csv")
        df.to_csv(output_path, mode='a' if os.path.exists(output_path) else 'w', 
                 header=not os.path.exists(output_path), index=False)
        log.debug(f"Saved optimized tube {original_id} to {output_path}")

def optimize_tube(files_pattern: str = "*/*.csv", output_dir: str = '../optimized_tubes',
                 video_length: int = 1474, epochs: int = 1000) -> None:
    # Device selection with multi-GPU support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs")
        device = "cuda:0"  # Use first GPU in multi-GPU setup

    log.info(f"🚀 Starting optimization on device: {device.upper()}")
    
    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Data loading and optimization
    tube_data, areas, boxes, original_tube_map, full_dataframes = load_tubes(files_pattern)
    optimized_starts = optimize_tube_starts(tube_data, areas, boxes, video_length, epochs, device=device)

    # Save results
    save_optimized_tubes(full_dataframes, optimized_starts, original_tube_map, output_dir)
    log.info(f"✅ Optimization complete. Results saved to {output_dir}")

    # Visualization
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    plot_tubes(ax1, tube_data, None, areas, 'Initial Arrangement')
    ax2 = fig.add_subplot(122, projection='3d')
    plot_tubes(ax2, tube_data, list(optimized_starts.values()), areas, 'Optimized Arrangement')
    plt.savefig('./optimized_tubes_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    log.info("📊 Visualization saved to optimized_tubes_plot.png")

if __name__ == "__main__":
    optimize_tube()
import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, deque

def blend_roi_on_background(background, roi, roi_mask, x1, x2, y1, y2):
    # Clamping ROI coordinates to ensure they are within the bounds of the background image
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, background.shape[1]), min(y2, background.shape[0])

    if x2 <= x1 or y2 <= y1:
        print("Adjusted ROI coordinates result in non-positive area.")
        return background  # Return unchanged background if ROI area is non-positive

    roi_width, roi_height = x2 - x1, y2 - y1
    roi_resized = cv2.resize(roi, (roi_width, roi_height))
    mask_resized = cv2.resize(roi_mask, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)

    # Convert mask to binary and ensure it is CV_8U
    _, mask_binary = cv2.threshold(mask_resized, 1, 255, cv2.THRESH_BINARY)
    if mask_binary.dtype != np.uint8:
        mask_binary = mask_binary.astype(np.uint8)

    roi_region = background[y1:y2, x1:x2]
    if roi_resized.shape[:2] != roi_region.shape[:2]:
        print("ROI resized shape does not match the ROI region shape.")
        return background

    # Apply the mask
    mask_3ch = cv2.merge([mask_binary] * 3)  # Ensure mask is 3 channels if needed
    roi_fg = cv2.bitwise_and(roi_resized, roi_resized, mask=mask_binary)
    roi_bg = cv2.bitwise_and(roi_region, roi_region, mask=cv2.bitwise_not(mask_binary))
    blended_roi = cv2.add(roi_fg, roi_bg)
    background[y1:y2, x1:x2] = blended_roi

    return background

def load_new_times(directory):
    new_times = defaultdict(dict)
    for filename in sorted(glob.glob(f"{directory}/*.txt")):
        with open(filename, "r") as file:
            for line in file:
                Tube, n, _, _, _, _, curr_time, *_  = line.strip().split(',')
                new_times[int(Tube)][int(n)] = float(curr_time)
    return new_times

def temporal_smoothing(tube_queue, time_window=3):
    """
    Applies temporal smoothing to the tube_queue to determine the presence of objects
    over a defined time window.
    """
    smoothed_queue = defaultdict(list)
    times = sorted(tube_queue.keys())
    for i, curr_time in enumerate(times):
        start_idx = max(0, i - time_window + 1)
        relevant_times = times[start_idx:i+1]
        count = {}
        # Collect occurrences of each object across the window
        for t in relevant_times:
            for data in tube_queue[t]:
                key = (data[0], data[1])  # Tube and n identifiers
                if key not in count:
                    count[key] = []
                count[key].append(data[2:])
        # Determine majority presence and average coordinates
        for key, entries in count.items():
            if len(entries) >= (time_window // 2) + 1:
                # Average out the coordinates
                avg_x1 = int(np.mean([e[0] for e in entries]))
                avg_x2 = int(np.mean([e[1] for e in entries]))
                avg_y1 = int(np.mean([e[2] for e in entries]))
                avg_y2 = int(np.mean([e[3] for e in entries]))
                smoothed_queue[curr_time].append((key[0], key[1], avg_x1, avg_x2, avg_y1, avg_y2))
    return smoothed_queue

def Tube_mod(args, video, bgimg=None, mask_dir='../masks', dir1 = "*", dir2 = "../optimized_tubes"):
    new_times = load_new_times(dir2)
    if bgimg is None:
        bgimg = np.asarray(Image.open('../bg/background_img.png'))
        bgimg = cv2.cvtColor(bgimg, cv2.COLOR_RGB2BGR)

    tube_queue = defaultdict(list)

    filenames = sorted(glob.glob(f"{dir1}/*.txt"))
    for filename in tqdm(filenames):
        with open(filename) as file:
            while (line := file.readline().rstrip('')):
                Tube, n, x1, x2, y1, y2, curr_time, *_ = line.split(',')
                Tube, n, x1, x2, y1, y2 = int(Tube), int(n), int(x1), int(x2), int(y1), int(y2)
                curr_time = new_times[Tube].get(n, float(curr_time))  # Use new time if available, else original

                tube_queue[curr_time].append((Tube, n, x1, x2, y1, y2))

    # Process tubes sorted by time
    for curr_time in sorted(tube_queue.keys()):
        combined_background = bgimg.copy() 

        for Tube, n, x1, x2, y1, y2 in tube_queue[curr_time]:
            image_path = f'{str(Tube).zfill(4)}/{str(n).zfill(4)}' + args['ext']
            mask_path = f'{str(mask_dir).zfill(4)}/{str(Tube).zfill(4)}/{str(n).zfill(4)}' + args['ext']

            image = np.asarray(Image.open(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mask = np.asarray(Image.open(mask_path))

            if image.shape[:2] != mask.shape[:2]:
                print(f"ROI {image.shape[:2]} and mask {mask.shape[:2]} dimensions do not match")

            temp_clone = blend_roi_on_background(combined_background, image, mask, x1, x2, y1, y2)
            combined_background[y1:y2, x1:x2] = temp_clone[y1:y2, x1:x2]

        background = combined_background.copy()

        if Tube > 0:
            video.write(background) 


def check_overlap(x1, x2, y1, y2, other_x1, other_x2, other_y1, other_y2):
    """ Calculate the overlap between two bounding boxes. """
    # Calculate the intersection
    xi1 = max(x1, other_x1)
    yi1 = max(y1, other_y1)
    xi2 = min(x2, other_x2)
    yi2 = min(y2, other_y2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    # Calculate the union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (other_x2 - other_x1) * (other_y2 - other_y1)
    union_area = box1_area + box2_area - inter_area
    # Calculate the IoU (Intersection over Union)
    iou = inter_area / union_area
    return iou > 0.1  # Allow for up to 10% overlap

def resize_tube(image, mask, x1, x2, y1, y2, min_size=50, reduction=2):
    """Resize the tube to reduce its dimensions slightly."""
    new_width = max((x2 - x1) // reduction, min_size)
    new_height = max((y2 - y1) // reduction, min_size)
    if new_width != (x2 - x1) or new_height != (y2 - y1):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_AREA)
        x1 += (x2 - x1 - new_width) // 2
        x2 = x1 + new_width
        y1 += (y2 - y1 - new_height) // 2
        y2 = y1 + new_height
    return image, mask, x1, x2, y1, y2

def create_composite_image(tube_queues, bg_shape):
    composite_image = np.zeros(bg_shape, dtype=np.uint8)
    composite_mask = np.zeros(bg_shape[:2], dtype=bool)

    for Tube, frames in list(tube_queues.items()):
        if frames:
            _, x1, x2, y1, y2, image, mask = frames[0]
            for other_Tube, other_frames in list(tube_queues.items()):
                if other_Tube != Tube and other_frames:
                    _, other_x1, other_x2, other_y1, other_y2, other_image, other_mask = other_frames[0]
                    if check_overlap(x1, x2, y1, y2, other_x1, other_x2, other_y1, other_y2):
                        image, mask, x1, x2, y1, y2 = resize_tube(image, mask, x1, x2, y1, y2)
                        other_image, other_mask, other_x1, other_x2, other_y1, other_y2 = resize_tube(other_image, other_mask, other_x1, other_x2, other_y1, other_y2)
            composite_image[y1:y2, x1:x2] = np.where(mask[:, :, None], image, composite_image[y1:y2, x1:x2])
            composite_mask[y1:y2, x1:x2] = mask.astype(bool)
            frames.popleft()

    return composite_image, composite_mask

def create_composite_image_size_chagne(tube_queues, bg_shape, log):
    composite_image = np.zeros(bg_shape, dtype=np.uint8)
    composite_mask = np.zeros(bg_shape[:2], dtype=bool)
    processed = set()
    active_tubes = [(Tube, frames[0]) for Tube, frames in tube_queues.items() if frames]
    overlap_checked = set()

    for i, (Tube, frame) in enumerate(active_tubes):
        x1, x2, y1, y2, image, mask = frame[1:]
        original_image = image.copy()
        original_mask = mask.copy()

        # Process overlaps and resize
        for j, (other_Tube, other_frame) in enumerate(active_tubes):
            if i != j and frozenset({Tube, other_Tube}) not in overlap_checked:
                overlap_checked.add(frozenset({Tube, other_Tube}))
                ox1, ox2, oy1, oy2, other_image, other_mask = other_frame[1:]
                
                if check_overlap(x1, x2, y1, y2, ox1, ox2, oy1, oy2):
                    image, mask, x1, x2, y1, y2 = resize_tube(image, mask, x1, x2, y1, y2)
                    other_image, other_mask, ox1, ox2, oy1, oy2 = resize_tube(other_image, other_mask, ox1, ox2, oy1, oy2)

        if Tube not in processed:
            # Final safety checks
            x1, x2 = sorted((max(0, x1), min(bg_shape[1], x2)))
            y1, y2 = sorted((max(0, y1), min(bg_shape[0], y2)))
            
            # Calculate actual region dimensions
            region_height = y2 - y1
            region_width = x2 - x1
            
            # Handle empty regions
            if region_height <= 0 or region_width <= 0:
                tube_queues[Tube].popleft()
                continue

            # Ensure image dimensions match region
            if image.shape[:2] != (region_height, region_width):
                image = cv2.resize(image, (region_width, region_height))
                mask = cv2.resize(mask.astype(np.uint8), (region_width, region_height))

            # Dimension alignment
            if image.ndim == 2:
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask.squeeze()
            
            # Final validation
            try:

                composite_image[y1:y2, x1:x2] = np.where(mask[..., None], image, composite_image[y1:y2, x1:x2])
                composite_mask[y1:y2, x1:x2] |= mask.astype(bool)
            except ValueError as e:
                # Fallback to original dimensions
                log.warning(f"[Error] {e}")
                composite_image[y1:y2, x1:x2] = np.where(original_mask[..., None], original_image, composite_image[y1:y2, x1:x2])
                composite_mask[y1:y2, x1:x2] |= original_mask.astype(bool)

            processed.add(Tube)
            tube_queues[Tube].popleft()

    return composite_image, composite_mask

def blend_on_background(background, composite_image, composite_mask):
    return np.where(composite_mask[:, :, None], composite_image, background)

def process_tube_frame(Tube, n, x1, x2, y1, y2, mask_dir):
    image_path = f'{str(Tube).zfill(4)}/{str(n).zfill(4)}.png'
    mask_path = f'{mask_dir}/{str(Tube).zfill(4)}/{str(n).zfill(4)}.png'
    image = np.asarray(Image.open(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    return image, mask, (x1, x2, y1, y2)

def load_tube_data(dir2, mask_dir):
    tube_queues = defaultdict(deque)
    filenames = sorted(glob.glob(dir2))
    for filename in tqdm(filenames, desc=u'⏳ Loading tube data'):
        with open(filename) as file:
            for line in file:
                if line.strip():
                    line = line.split(',')
                    Tube, n, x1, x2, y1, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]), int(line[5])
                    image, mask, coords = process_tube_frame(Tube, n, x1, x2, y1, y2, mask_dir)
                    tube_queues[Tube].append((n, *coords, image, mask))
    return tube_queues

def convert_directory_csv_to_txt(input_dir:str, output_dir:str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(input_dir, filename)
            txt_filename = filename.replace('.csv', '.txt')
            txt_file_path = os.path.join(output_dir, txt_filename)
            csv_to_txt(csv_file_path, txt_file_path)

def csv_to_txt(csv_file_path:str, txt_file_path:str):
    df = pd.read_csv(csv_file_path)
    tube_id = csv_file_path.split('/')[-1].split('.')[0].split('_')[-1].zfill(4)
    df['T'] = tube_id
    df['time'] = df['time']
    
    output_columns = ['T', 'n', 'x1', 'x2', 'y1', 'y2', 'time']
    if 'n' not in df.columns:
        df['n'] = range(1, len(df) + 1)
    
    formatted_data = df[output_columns].apply(
        lambda x: f"{x['T']}, {x['n']}, {x['x1']}, {x['x2']}, {x['y1']}, {x['y2']}, {x['time']:.2f},",
        axis=1
    )
    
    with open(txt_file_path, 'w') as f:
        for line in formatted_data:
            f.write(line + '\n')

def Tube(args: dict, 
         video: cv2.VideoWriter,  
         bgimg:np.array=None, 
         mask_dir:str='../masks', 
         dir2:str="../optimized_tubes/*.txt",
         log=None):
    tube_queues = load_tube_data(dir2, mask_dir)
    if bgimg is None:
        bgimg = np.asarray(Image.open(args['bg_path']))
        bgimg = cv2.cvtColor(bgimg, cv2.COLOR_RGB2BGR)

    max_frames = max(len(frames) for frames in tube_queues.values())
    for _ in tqdm(range(max_frames), desc=u'⏳ Processing frames'):
        composite_image, composite_mask = create_composite_image_size_chagne(tube_queues, bgimg.shape, log)
        current_bg = blend_on_background(np.copy(bgimg), composite_image, composite_mask)
        video.write(current_bg)

    video.release()
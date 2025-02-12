import os
import csv
import time
import shutil
import logging
import datetime

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from torch.amp import autocast
import albumentations as albu

from sort import Sort
from utils import pad_image, pad_to_max_shape
from energy import optimize_tube
from tube_util import Tube, convert_directory_csv_to_txt
from people_segmentation.pre_trained_models import create_model
from supplementary.our_args import args, save_yaml_config, CONFIG_PATH
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image

def run_inference_on_batch(frames: List[np.ndarray], args: dict) -> List[np.ndarray]:
    """
    Runs model inference on a batch of frames and returns binary masks.

    Args:
        frames (List[np.ndarray]): List of frames.
        args (dict): Configuration arguments.

    Returns:
        List[np.ndarray]: List of binary masks.
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = create_model(args['input_model']).to(device)
    batch_size = args['batch_size']

    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs!")
        batch_size *= torch.cuda.device_count()
        model = torch.nn.DataParallel(model)
    
    model.eval()
    transform = albu.Compose([albu.Normalize(p=1)], p=1)

    transformed_batch = []
    for frame in frames:
        transformed = transform(image=frame)["image"]
        padded_frame, _ = pad_image(transformed)
        transformed_batch.append(tensor_from_rgb_image(padded_frame))

    x = torch.stack(transformed_batch).to(device)

    with torch.no_grad():
        if device == 'cuda':
            with autocast(device):
                predictions = model(x)
        else:
            predictions = model(x)

    return [(pred[0].cpu().numpy() > 0).astype(np.uint8) for pred in predictions]


def generate_tubes_online(args: dict, cap: cv2.VideoCapture, video_length: int, fps: int=15) -> int:
    """
    Perform online tube generation:  
    - Read frames, preprocess, run inference via `run_inference_on_batch`, track objects  
    - Save object regions and metadata  

    Returns:
        int: Total image count processed.
    """
    log.info(f'[Info] Using device: {"CUDA" if torch.cuda.is_available() else "CPU"} ✅')

    tracker = Sort(max_age=3, min_hits=3, iou_threshold=0.3)
    pbar = tqdm(total=video_length // args['batch_size'], desc='⏳ Processing Batches', unit='batch')

    image_number = 1
    next_available_id = 1
    object_id_mapping = {}

    frame_index = 0
    height, width = None, None

    while True:
        frames_batch = []
        frame_indices = []

        for _ in range(args['batch_size']):
            ret, frame = cap.read()
            if not ret:
                break
            if height is None:
                height, width, _ = frame.shape
            frames_batch.append(frame)
            frame_indices.append(frame_index)
            frame_index += 1

        if not frames_batch:
            break

        # Run inference to get binary masks
        binary_masks = run_inference_on_batch(frames_batch, args)

        for i, (mask, original) in enumerate(zip(binary_masks, frames_batch)):
            current_time = frame_indices[i] / fps

            if mask.size == 0:
                continue

            mask = cv2.resize(mask, (width, height))
            mask_e = mask * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            detections = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if cv2.contourArea(contour) > (width * height) // 90:
                    detections.append([x, y, x + w, y + h])

            if detections:
                tracked_objects = tracker.update(np.array(detections))

                for track in tracked_objects:
                    objectID = int(track[4])
                    coords = [max(0, int(coord)) for coord in track[:4]]

                    if objectID not in object_id_mapping:
                        object_id_mapping[objectID] = next_available_id
                        next_available_id += 1

                    new_id = object_id_mapping[objectID]
                    ROI = original[coords[1]:coords[3], coords[0]:coords[2]]
                    mask_roi = mask_e[coords[1]:coords[3], coords[0]:coords[2]]

                    if ROI.size == 0:
                        continue

                    TubeID = str(new_id).zfill(4)
                    curr_time_str = f'{current_time:.2f}'
                    os.makedirs(TubeID, exist_ok=True)
                    os.makedirs(f'../masks/{TubeID}', exist_ok=True)

                    cv2.imwrite(f'{TubeID}/{str(image_number).zfill(4)}{args["ext"]}', ROI)
                    cv2.imwrite(f'../masks/{TubeID}/{str(image_number).zfill(4)}{args["ext"]}', mask_roi)

                    with open(f'{TubeID}/{TubeID}node.txt', 'a') as f:
                        f.write(f'{TubeID}, {image_number}, {coords[0]}, {coords[2]}, {coords[1]}, {coords[3]}, {curr_time_str},\n')

                    with open(f'{TubeID}/{TubeID}node.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if csvfile.tell() == 0:
                            writer.writerow(['T', 'n', 'x1', 'y1', 'x2', 'y2', 'time', 'contour'])
                        writer.writerow([int(TubeID), image_number, *coords, curr_time_str, contour])

                    image_number += 1

        pbar.update(1)
        if image_number > video_length:
            log.info(f'[Info] image_number: {image_number} exceeds video_length: {video_length}')
            break

    pbar.close()
    cap.release()
    return image_number


def optimize_tubes(args: dict, 
                   video: cv2.VideoWriter, 
                   bgimg: np.ndarray, 
                   final: np.ndarray, 
                   video_length: int, 
                   epochs: int):
    """
    Optimize tubes using either MCTS or Energy Optimization and stitch them into the final video.
    """
    log.info('[Info] Optimizing tubes with Energy Optimization...')
    optimize_tube(
        files_pattern=args['files_csv_dir'],
        output_dir=args['optimized_tubes_dir'],
        video_length=video_length,
        epochs=epochs
    )
    convert_directory_csv_to_txt(args['optimized_tubes_dir'], args['optimized_tubes_dir'])
    Tube(args, video, bgimg=bgimg, final=final, dir2=f"{args['optimized_tubes_dir']}/*.txt")

def main(args: dict, 
         cap: cv2.VideoCapture, 
         video: cv2.VideoWriter, 
         video_length: int, 
         final: np.ndarray, 
         bgimg: np.ndarray, 
         energy_opt: bool = True, 
         epochs: int = 1000, 
         final_video_name: str = None, 
         size: tuple = (1080, 1920)):
    """
    Main function to orchestrate online tube generation and optimization.
    """
    start_time = time.time()
    
    image_count = generate_tubes_online(args, cap, video_length)
    
    if energy_opt:
        optimize_tubes(args, video, bgimg, final, video_length, epochs)
    else:
        Tube(args, video, bgimg=bgimg, final=final, dir2="*/*.txt")
    
    log.info(f'[Info] Video saved at {final_video_name} ✅')
    log.info(f'⏳ Total time: {time.time() - start_time:.2f}s ⏳')
    log.info('[Finish 🙌 🏁]...')

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("video_synopsis.log"),  
            logging.StreamHandler() 
            ]
        )
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    
    save_yaml_config(args, CONFIG_PATH)
    log.info(args)

    # Set paths
    output_path = args["output"]
    final = args["masks"]
    synopsis_frames = args["synopsis_frames"]
    energy_opt = args["energy_optimization"]
    epochs = args["epochs"]

    # Create or clear directories
    def prepare_directory(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

    for path in [output_path, synopsis_frames, final]:
        prepare_directory(path)
        if str(path) is str(output_path): os.chdir(output_path) 

    # Configure background subtraction
    fgbg = cv2.createBackgroundSubtractorKNN(127, cv2.THRESH_BINARY, 1)
    fgbg.setDetectShadows(False)

    # Video capture configuration
    video_path = args["video"]
    cap = cv2.VideoCapture(video_path)      
    cap1 = cv2.VideoCapture(video_path)     

    if not cap.isOpened() or not cap1.isOpened():
        log.error(f"[Error]: Unable to open video file {video_path}")
        raise RuntimeError(f"[Error]: Unable to open video file {video_path}")

    # Obtain video properties
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    video_length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Output video properties
    log.info(f"[Original Video] Frame Width: {frame_width}, Frame Height: {frame_height} ✅")
    log.info(f"[Original Video] Total Frames: {video_length} ✅")
    log.info(f"[Original Video] FPS: {fps} ✅")

    # Random frame selection for background median
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    rand_ids = np.random.choice(total_frames, size=fps, replace=False)
    sampled_frames = []

    for frame_id in rand_ids:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap1.read()
        if ret and frame is not None:
            sampled_frames.append(frame)

    # Ensure all frames are padded to the same size
    padded_frames = pad_to_max_shape(sampled_frames)

    # Compute median frame and save it
    median_frame = np.median(padded_frames, axis=0).astype(np.uint8)
    bg_path = args["bg_path"]
    bg_dir = os.path.dirname(bg_path)

    # Ensure the directory exists
    if not os.path.exists(bg_dir):
        os.makedirs(bg_dir)  
    cv2.imwrite(bg_path, median_frame)

    # Preprocess median frame
    gray_median = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    smooth_median = cv2.GaussianBlur(gray_median, (5, 5), 0)

    # Load and prepare background image
    bgimg = np.asarray(Image.open(bg_path))
    bgimg = cv2.cvtColor(bgimg, cv2.COLOR_RGB2BGR)

    # Video writer setup
    if frame_width > 0 and frame_height > 0:
        video_name = f"../{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.mp4"
        video = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
        )

        if not video.isOpened():
            log.error("[Error]: Could not open video writer")
            raise RuntimeError("[Error]: Could not open video writer. ❌")
    else:
        log.error("[Error]: Invalid frame dimensions")
        raise ValueError("[Error]: Invalid frame dimensions. ❌")

    # Main processing
    main(args, cap, video, video_length, final, bgimg, energy_opt, epochs, video_name,)
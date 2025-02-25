import argparse
import yaml
import os

CONFIG_PATH = "config.yaml"

def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {} 
    return {}

def save_yaml_config(config, config_path):
    """Save configuration to a YAML file."""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

parser = argparse.ArgumentParser(description="Process video and perform energy optimization for video synopsis.")

parser.add_argument(
    "-b", "--buff_size", 
    help="Buffer size for capturing footage (default: 32 frames)", 
    type=int, 
    default=32
)
parser.add_argument(
    "-v", "--video", 
    help="Path to the input video file (default: '../src/all_rush_video.mp4')", 
    type=str,
    default="../src/all_rush_video.mp4"
)
parser.add_argument(
    "-inmod", "--input_model", 
    help="Path to the input model (default: 'Unet_2020-07-20')", 
    type=str,
    default="Unet_2020-07-20"
)
parser.add_argument(
    "-e", "--ext", 
    help="File extension for extracted objects (default: '.png')", 
    type=str,
    default=".png"
)
parser.add_argument(
    "-cv", "--dvalue", 
    help="Compression value (0-9; default: 9)", 
    type=int, 
    default=9
)
parser.add_argument(
    "-f", "--fps", 
    help="Frames per second (FPS) for video processing (default: 15)", 
    type=int, 
    default=25
)
parser.add_argument(
    "-bsz", "--batch_size", 
    help="Batch size for frame wise inference (default: 8)", 
    type=int, 
    default=8
)
parser.add_argument(
    "--files_csv_dir", 
    help="Directory to save tube CSV files (default: '*/*.csv')", 
    type=str, 
    default="*/*.csv"
)
parser.add_argument(
    "--optimized_tubes_dir", 
    help="Directory for optimized tubes (default: '../optimized_tubes')", 
    type=str, 
    default="../optimized_tubes"
)
parser.add_argument(
    "--output", 
    help="Output directory for processed data (default: 'output')", 
    type=str, 
    default="output"
)
parser.add_argument(
    "--masks", 
    help="Directory containing masks (default: '../masks')", 
    type=str, 
    default="../masks"
)
parser.add_argument(
    "--synopsis_frames", 
    help="Directory for synopsis frames (default: '../synopsis_frames')", 
    type=str, 
    default="../synopsis_frames"
)
parser.add_argument(
    "--energy_optimization", 
    help="Enable or disable energy optimization (default: True)", 
    type=bool, 
    default=True
)

parser.add_argument(
    "--epochs", 
    help="Number of epochs for energy optimization (default: 2000)", 
    type=int, 
    default=2000
)
parser.add_argument(
    "--bg_path", 
    help="Path to the extracted background image (default: '../bg/background_img.png')", 
    type=str, 
    default="../bg/background_img.png"
)
parser.add_argument(
    "--sigma", 
    help="Sigma value for Gaussian blur (default: 1.5)",
    type=float, 
    default=50.0
)

parser.add_argument(
    "--collision_method",
    help="Method to handle collisions (default: 'repulsion')", 
    type=str, 
    default='repulsion',
)

args = vars(parser.parse_args())

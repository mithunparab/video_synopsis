"""Backward-compatible entry point. Delegates to the new video_synopsis package.

Usage:
    python main.py -v /path/to/video.mp4 --epochs 2000

For the new API, use:
    python -m video_synopsis.cli -v /path/to/video.mp4
"""

import logging
import sys
import os

# Allow importing the old modules for backward compatibility
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_synopsis.config import Config
from video_synopsis.pipeline import Pipeline


def main():
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    log = logging.getLogger(__name__)

    config = Config.from_args()
    config.save_yaml("config.yaml")
    log.info(f"Config: {config.to_dict()}")

    pipeline = Pipeline(config)
    output = pipeline.run()

    if output:
        log.info(f"Video saved at {output}")
    log.info("Finished.")


if __name__ == "__main__":
    main()

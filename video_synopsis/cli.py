"""Thin CLI wrapper for the video synopsis pipeline."""

import logging
import sys

from video_synopsis.config import Config
from video_synopsis.pipeline import Pipeline


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    config = Config.from_args()
    config.save_yaml("config.yaml")

    pipeline = Pipeline(config)
    output = pipeline.run()

    if output:
        print(f"Output video: {output}")
    else:
        print("No output generated.")


if __name__ == "__main__":
    main()

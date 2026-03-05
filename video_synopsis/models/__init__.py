"""Pluggable segmentation and tracking models."""

from video_synopsis.models.base import BaseSegmenter, BaseTracker, SegmentationResult

__all__ = ["BaseSegmenter", "BaseTracker", "SegmentationResult"]

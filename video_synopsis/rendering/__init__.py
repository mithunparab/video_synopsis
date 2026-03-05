"""Video rendering and compositing."""


def __getattr__(name):
    if name == "extract_background":
        from video_synopsis.rendering.background import extract_background
        return extract_background
    if name == "Stitcher":
        from video_synopsis.rendering.stitcher import Stitcher
        return Stitcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["extract_background", "Stitcher"]

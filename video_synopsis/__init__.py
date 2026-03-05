"""Video Synopsis: condense surveillance videos into short summaries."""


def __getattr__(name):
    if name == "Pipeline":
        from video_synopsis.pipeline import Pipeline
        return Pipeline
    if name == "Config":
        from video_synopsis.config import Config
        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Pipeline", "Config"]

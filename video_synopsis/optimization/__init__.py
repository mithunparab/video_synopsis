"""Tube placement optimization."""


def __getattr__(name):
    if name == "BaseOptimizer":
        from video_synopsis.optimization.base import BaseOptimizer
        return BaseOptimizer
    if name == "compute_pairwise_collision_3d":
        from video_synopsis.optimization.collision import compute_pairwise_collision_3d
        return compute_pairwise_collision_3d
    if name == "EnergyOptimizer":
        from video_synopsis.optimization.energy import EnergyOptimizer
        return EnergyOptimizer
    if name == "MCTSOptimizer":
        from video_synopsis.optimization.mcts import MCTSOptimizer
        return MCTSOptimizer
    if name == "PSOOptimizer":
        from video_synopsis.optimization.pso import PSOOptimizer
        return PSOOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseOptimizer",
    "compute_pairwise_collision_3d",
    "EnergyOptimizer",
    "MCTSOptimizer",
    "PSOOptimizer",
]

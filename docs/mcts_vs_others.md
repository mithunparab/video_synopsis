# MCTS vs Energy / PSO — where MCTS actually wins

Three structural wins, independent of self-play training:

1. **Hard-constraint placement.** MCTS picks order + slot via `_find_earliest_start_time` (`video_synopsis/optimization/mcts.py:179`), so its output has *zero* spatial-temporal overlap by construction. Energy and PSO minimize a soft collision penalty — there's always residual overlap, which shows as ghosting in the stitched video. If the evaluation metric is "visible collisions," MCTS wins by definition.

2. **Bin-packing with heterogeneous durations.** When tubes vary wildly in length (5 frames vs 800), gradient/swarm methods get pulled around by the long ones and waste the small gaps. MCTS's slot-fit naturally drops short tubes into leftover holes — tighter compression ratio on the same input.

3. **Strongly clustered scenes.** When many tubes share the same spatial region (doorway, corridor, ATM), the energy landscape is full of bad local minima — gradient/PSO converge to whichever ordering they happened to start from. MCTS lookahead picks a non-greedy ordering (place the longest blocker first, fill around it). This is where the gap is widest.

## Where MCTS loses (even with self-play)

- Few tubes (<20): wall-clock and often final answer.
- Spatially disjoint scenes (no collisions to resolve).
- Sub-frame timing precision — energy opt's continuous relaxation handles that better and runs in seconds.

## Honest framing

The pitch isn't "MCTS is better given enough compute" — it's:

> MCTS is the right tool when collisions are *the* bottleneck and a hard zero-overlap guarantee is required.

Otherwise, energy optimization is faster, simpler, and competitive.

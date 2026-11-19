"""Per-point labels from the .route file.

.route lists arcs in driving order but has no timestamps, and has far fewer rows than
.track. Snapping each point to its nearest route arc breaks where the route loops back
over itself, so we solve a monotone alignment with dynamic programming:

    C[i, j] = d(point_i, route_arc_j) + min_{k <= j} C[i-1, k]

i.e. as time moves forward the position along the route may only stay or advance.
The best path gives each GPS point one route arc = its label.
"""
import numpy as np

from .geometry import project_to_segments
from .config import LABEL_MAX_DIST_M


def align_points_to_route(xy, node_xy, arcs, route):
    A, B = node_xy[arcs[route, 0]], node_xy[arcs[route, 1]]
    n, m = len(xy), len(route)
    back = np.empty((n, m), dtype=np.int16 if m < 32_000 else np.int32)
    prev = None
    for i in range(n):
        d, _ = project_to_segments(np.broadcast_to(xy[i], (m, 2)), A, B)
        if prev is None:
            cur = d
            back[i] = np.arange(m)
        else:
            # prefix minimum of the previous row, and the index where that minimum sits
            better = np.minimum.accumulate(prev)
            idx = np.maximum.accumulate(np.where(prev == better, np.arange(m), 0))
            back[i] = idx
            cur = d + better
        prev = cur
    # backtrack
    j = int(np.argmin(prev))
    pos = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = int(back[i, j])
    label_arc = route[pos]
    dist, _ = project_to_segments(xy, A[pos], B[pos])
    valid = dist <= LABEL_MAX_DIST_M
    return label_arc, pos, dist, valid

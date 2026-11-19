"""Point-to-segment geometry and the spatial index used for candidate generation."""
import numpy as np
from scipy.spatial import cKDTree

from .config import DENSIFY_STEP_M


def project_to_segments(P, A, B):
    """Distance from points P to segments A->B (row-wise) and the projection
    parameter t in [0,1] (0 = start node, 1 = end node)."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    t = np.clip(((P - A) * AB).sum(1) / L2, 0.0, 1.0)
    proj = A + t[:, None] * AB
    return np.linalg.norm(P - proj, axis=1), t


class ArcIndex:
    """KD-tree over points sampled every DENSIFY_STEP_M along every arc.
    Long arcs are covered properly (a midpoint-only index misses them)."""

    def __init__(self, node_xy, arcs):
        self.A = node_xy[arcs[:, 0]]
        self.B = node_xy[arcs[:, 1]]
        self.length = np.linalg.norm(self.B - self.A, axis=1)
        n_samples = np.maximum(1, np.ceil(self.length / DENSIFY_STEP_M).astype(int)) + 1
        arc_of_sample = np.repeat(np.arange(len(arcs)), n_samples)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_samples])
        pts = self.A[arc_of_sample] + frac[:, None] * (self.B - self.A)[arc_of_sample]
        self.arc_of_sample = arc_of_sample
        self.tree = cKDTree(pts)

    def candidates(self, xy, radius, k_max):
        """For each point: arrays (arc_ids, dist, t) of the k_max closest arcs within radius."""
        hits = self.tree.query_ball_point(xy, r=radius + DENSIFY_STEP_M)
        out = []
        for p, h in zip(xy, hits):
            if not h:
                out.append((np.empty(0, int), np.empty(0), np.empty(0)))
                continue
            ids = np.unique(self.arc_of_sample[h])
            d, t = project_to_segments(np.broadcast_to(p, (len(ids), 2)), self.A[ids], self.B[ids])
            keep = d <= radius
            ids, d, t = ids[keep], d[keep], t[keep]
            order = np.argsort(d, kind="stable")[:k_max]
            out.append((ids[order], d[order], t[order]))
        return out


def bearing_vectors(xy):
    """Unit direction of travel from central differences (forward/backward at the ends)."""
    v = np.gradient(xy, axis=0)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-9), n[:, 0]

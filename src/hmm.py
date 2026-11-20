"""HMM + Viterbi route reconstruction.

Hidden states   : at GPS point i, the state is "the vehicle is on candidate arc c" (the K
                  candidates from candidate generation, so state <-> arc_id is kept explicitly).
Emission        : log P(classifier says candidate c is correct) - the selected model's output.
Transition      : Newson & Krumm (2009) style. With d_gc = straight-line distance between the two
                  GPS points and d_route = network distance from the projection on arc a to the
                  projection on arc b:  log P(b | a) = -|d_route - d_gc| / beta
                  Unreachable pairs (no directed path within the search cutoff, e.g. a
                  disconnected component) get a large finite penalty, so the chain never dies.
GPS gaps        : if dt > HMM_GAP_S the chain is split and each part is decoded independently.
No candidates   : the point gets no prediction (-1) and is skipped by the chain.
"""
import numpy as np
import networkx as nx

from .config import HMM_BETA_M, HMM_GAP_S, HMM_UNREACHABLE_LOGP, HMM_EMISSION_FLOOR


class RoadGraph:
    def __init__(self, node_xy, arcs, cutoff_m=600.0):
        self.arcs = arcs
        self.length = np.linalg.norm(node_xy[arcs[:, 1]] - node_xy[arcs[:, 0]], axis=1)
        self.G = nx.DiGraph()
        self.G.add_weighted_edges_from(zip(arcs[:, 0].tolist(), arcs[:, 1].tolist(), self.length.tolist()))
        self.cutoff = cutoff_m
        self._cache = {}

    def sp_from(self, src, need):
        """Shortest network distances from node src, exploring up to max(cutoff, need) metres."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= need:
            return hit[1]
        lim = max(self.cutoff, need)
        d = nx.single_source_dijkstra_path_length(self.G, src, cutoff=lim, weight="weight")
        self._cache[src] = (lim, d)
        return d


def transition_logp(graph, a_ids, a_t, b_ids, b_t, d_gc, beta):
    """Matrix (len(a), len(b)) of log transition probabilities."""
    L = graph.length
    need = 3 * d_gc + 150.0
    out = np.full((len(a_ids), len(b_ids)), HMM_UNREACHABLE_LOGP)
    for i, (a, ta) in enumerate(zip(a_ids, a_t)):
        to_a = graph.arcs[a, 1]
        dist = graph.sp_from(int(to_a), need)
        rem = (1.0 - ta) * L[a]
        for j, (b, tb) in enumerate(zip(b_ids, b_t)):
            if a == b:
                d_route = abs(tb - ta) * L[a]
            else:
                sp = dist.get(int(graph.arcs[b, 0]))
                if sp is None:
                    continue
                d_route = rem + sp + tb * L[b]
            out[i, j] = -abs(d_route - d_gc) / beta
    return out


def viterbi_trip(graph, xy, times, cand_by_obs, beta=HMM_BETA_M, gap_s=HMM_GAP_S, emission_weight=1.0):
    """cand_by_obs: dict obs -> (arc_ids, proj_t, prob). Returns predicted arc per obs (-1 if none)."""
    n = len(xy)
    pred = np.full(n, -1, dtype=np.int64)
    obs_seq = [i for i in range(n) if i in cand_by_obs and len(cand_by_obs[i][0])]
    # split into segments at large time gaps
    segments, cur = [], []
    for i in obs_seq:
        if cur and times[i] - times[cur[-1]] > gap_s:
            segments.append(cur); cur = []
        cur.append(i)
    if cur:
        segments.append(cur)

    for seg in segments:
        ids0, t0, p0 = cand_by_obs[seg[0]]
        score = emission_weight * np.log(np.maximum(p0, HMM_EMISSION_FLOOR))
        backs = []
        prev = seg[0]
        for i in seg[1:]:
            ids, tt, p = cand_by_obs[i]
            pa_ids, pa_t, _ = cand_by_obs[prev]
            d_gc = float(np.linalg.norm(xy[i] - xy[prev]))
            T = transition_logp(graph, pa_ids, pa_t, ids, tt, d_gc, beta)
            tot = score[:, None] + T                       # (prev states, current states)
            back = np.argmax(tot, axis=0)
            score = tot[back, np.arange(len(ids))] + emission_weight * np.log(np.maximum(p, HMM_EMISSION_FLOOR))
            backs.append(back)
            prev = i
        # backtrack
        k = int(np.argmax(score))
        for pos in range(len(seg) - 1, -1, -1):
            pred[seg[pos]] = cand_by_obs[seg[pos]][0][k]
            if pos > 0:
                k = int(backs[pos - 1][k])
    return pred

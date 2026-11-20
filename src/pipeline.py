"""End-to-end inference pipeline used by the API (and usable offline).

predict_trip()  : GPS + road network only  ->  candidates -> features -> classifier -> HMM/Viterbi
evaluate_trip() : compares predictions with the reference route. Runs server-side only, AFTER prediction.
The two are deliberately separate so the reference route can never influence a prediction.
"""
import time
import numpy as np
from scipy.spatial import cKDTree

from .config import HMM_BETA_M
from .features import FEATURES, build_trip_table
from .hmm import RoadGraph, viterbi_trip
from .labels import align_points_to_route


def predict_trip(trip, model, beta=HMM_BETA_M):
    assert trip.route is None, "predict_trip must receive a trip loaded with with_route=False"
    timings = {}
    t0 = time.perf_counter()
    cand, _ = build_trip_table(trip, with_labels=False)
    timings["candidates_features_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cand["prob"] = model.predict_proba(cand[FEATURES].values)[:, 1]
    timings["classifier_s"] = time.perf_counter() - t0

    n = len(trip.track)
    clf_pred = np.full(n, -1, dtype=np.int64)
    best = cand.loc[cand.groupby("obs").prob.idxmax()]
    clf_pred[best.obs.values] = best.arc_id.values

    t0 = time.perf_counter()
    graph = RoadGraph(trip.node_xy, trip.arcs)
    groups = {o: (g.arc_id.values, g.proj_t.values, g.prob.values) for o, g in cand.groupby("obs")}
    hmm_pred = viterbi_trip(graph, trip.xy, trip.track[:, 2], groups, beta=beta)
    timings["hmm_viterbi_s"] = time.perf_counter() - t0

    cand_sets = {o: set(v[0].tolist()) for o, v in groups.items()}
    return {"hmm_pred": hmm_pred, "clf_pred": clf_pred, "cand_sets": cand_sets,
            "n_candidate_rows": int(len(cand)), "timings": timings}


def continuity(arcs, pred):
    p = pred[pred >= 0]
    if len(p) < 2:
        return float("nan")
    a, b = p[:-1], p[1:]
    return float(((a == b) | (arcs[a, 1] == arcs[b, 0])).mean())


def route_set_scores(pred, route):
    p, r = set(pred[pred >= 0].tolist()), set(route.tolist())
    tp = len(p & r)
    prec, rec = tp / max(len(p), 1), tp / max(len(r), 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def evaluate_trip(trip_with_route, result):
    """Server-side scoring against the reference route (same definitions as scripts/train_models + run_hmm)."""
    tr = trip_with_route
    label_arc, _, _, valid = align_points_to_route(tr.xy, tr.node_xy, tr.arcs, tr.route)
    hmm, clf = result["hmm_pred"], result["clf_pred"]
    in_cands = np.array([label_arc[i] in result["cand_sets"].get(i, ()) for i in range(len(label_arc))])
    return {
        "hmm_top1_accuracy": float((hmm[valid] == label_arc[valid]).mean()),
        "classifier_top1_accuracy": float((clf[valid] == label_arc[valid]).mean()),
        "hmm_route_continuity": continuity(tr.arcs, hmm),
        "classifier_route_continuity": continuity(tr.arcs, clf),
        "hmm_route": route_set_scores(hmm, tr.route),
        "classifier_route": route_set_scores(clf, tr.route),
        "candidate_miss_rate": float(1 - in_cands[valid].mean()),
        "evaluated_points": int(valid.sum()),
        "points_without_reference": int((~valid).sum()),
    }


def route_polylines(trip, pred):
    """Consecutive unique predicted arcs -> list of [[lat, lon], [lat, lon]] segments for map drawing."""
    p = pred[pred >= 0]
    if len(p) == 0:
        return []
    p = p[np.r_[True, p[1:] != p[:-1]]]
    nd = trip.nodes
    return [[[nd[a, 1], nd[a, 0]], [nd[b, 1], nd[b, 0]]] for a, b in trip.arcs[p]]


def nearby_network(trip, radius_m=60.0, max_arcs=8000):
    """Road-network arcs near the trajectory (reverse twins drawn once) for map context."""
    near = cKDTree(trip.xy).query(trip.node_xy, distance_upper_bound=radius_m)[0] < np.inf
    ids = np.where(near[trip.arcs[:, 0]] & near[trip.arcs[:, 1]])[0]
    a = trip.arcs[ids]
    lo, hi = np.minimum(a[:, 0], a[:, 1]), np.maximum(a[:, 0], a[:, 1])
    _, first = np.unique(lo.astype(np.int64) * len(trip.nodes) + hi, return_index=True)
    a = a[np.sort(first)][:max_arcs]
    nd = trip.nodes
    return [[[nd[u, 1], nd[u, 0]], [nd[v, 1], nd[v, 0]]] for u, v in a]

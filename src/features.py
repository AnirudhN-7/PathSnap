"""Candidate generation + feature engineering.
One row = one (GPS point, candidate arc) pair; target y = 1 if the candidate is the labelled arc.
Only GPS + road-network information is used for features: labels never enter a feature."""
import numpy as np
import pandas as pd

from .config import CAND_RADIUS_M, MAX_CANDIDATES
from .geometry import ArcIndex, bearing_vectors, project_to_segments
from .labels import align_points_to_route

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "proj_t", "arc_len",
    "heading_cos", "heading_valid", "dist_prev_pt", "dist_next_pt",
    "has_reverse", "to_out_deg", "from_in_deg",
    "n_cand", "speed", "dt_prev", "dt_next", "turn_cos",
]


def trip_graph_stats(trip):
    n_nodes = len(trip.nodes)
    a = trip.arcs
    out_deg = np.bincount(a[:, 0], minlength=n_nodes)
    in_deg = np.bincount(a[:, 1], minlength=n_nodes)
    key = a[:, 0].astype(np.int64) * n_nodes + a[:, 1]
    rev_key = a[:, 1].astype(np.int64) * n_nodes + a[:, 0]
    has_reverse = np.isin(rev_key, key)
    return out_deg, in_deg, has_reverse


def build_trip_table(trip, with_labels=True):
    xy, nxy = trip.xy, trip.node_xy
    t = trip.track[:, 2]
    n = len(xy)
    idx = ArcIndex(nxy, trip.arcs)
    cands = idx.candidates(xy, CAND_RADIUS_M, MAX_CANDIDATES)
    out_deg, in_deg, has_reverse = trip_graph_stats(trip)

    unit, _ = bearing_vectors(xy)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.diff(t)
    dt_prev = np.r_[np.nan, dt]
    dt_next = np.r_[dt, np.nan]
    speed_step = step / np.maximum(dt, 1e-9)
    speed = np.r_[speed_step[:1], (speed_step[:-1] + speed_step[1:]) / 2, speed_step[-1:]] if n > 2 else np.zeros(n)
    sv = np.diff(xy, axis=0)
    sv = sv / np.maximum(np.linalg.norm(sv, axis=1, keepdims=True), 1e-9)
    turn = np.r_[1.0, (sv[:-1] * sv[1:]).sum(1), 1.0] if n > 2 else np.ones(n)

    if with_labels and trip.route is not None:
        label_arc, _, label_dist, label_valid = align_points_to_route(xy, nxy, trip.arcs, trip.route)
    else:
        label_arc = np.full(n, -1); label_dist = np.full(n, np.nan); label_valid = np.zeros(n, bool)

    rows = []
    for i, (ids, d, pt) in enumerate(cands):
        k = len(ids)
        if k == 0:
            continue
        A, B = idx.A[ids], idx.B[ids]
        av = (B - A) / np.maximum(idx.length[ids, None], 1e-9)
        hvalid = speed[i] > 1.0                     # heading is meaningless when standing still
        hcos = (av @ unit[i]) if hvalid else np.zeros(k)
        dp, _ = project_to_segments(np.broadcast_to(xy[max(i - 1, 0)], (k, 2)), A, B)
        dn, _ = project_to_segments(np.broadcast_to(xy[min(i + 1, n - 1)], (k, 2)), A, B)
        rows.append(pd.DataFrame({
            "obs": i, "t": t[i], "arc_id": ids,
            "dist": d, "dist_rank": np.arange(k), "dist_minus_min": d - d[0],
            "dist_ratio": d / (d[0] + 1.0), "proj_t": pt, "arc_len": idx.length[ids],
            "heading_cos": hcos, "heading_valid": float(hvalid),
            "dist_prev_pt": dp, "dist_next_pt": dn,
            "has_reverse": has_reverse[ids].astype(float),
            "to_out_deg": out_deg[trip.arcs[ids, 1]], "from_in_deg": in_deg[trip.arcs[ids, 0]],
            "n_cand": k, "speed": speed[i],
            "dt_prev": dt_prev[i], "dt_next": dt_next[i], "turn_cos": turn[i],
            "label_arc": label_arc[i], "label_valid": label_valid[i],
        }))
    df = pd.concat(rows, ignore_index=True)
    df["y"] = (df["arc_id"] == df["label_arc"]).astype(np.int8)
    df["dt_prev"] = df["dt_prev"].fillna(1.0)
    df["dt_next"] = df["dt_next"].fillna(1.0)
    df.insert(0, "trip", trip.trip_id)
    df.insert(1, "partition", trip.partition)

    point_info = pd.DataFrame({
        "trip": trip.trip_id, "partition": trip.partition, "obs": np.arange(n), "t": t,
        "lon": trip.track[:, 0], "lat": trip.track[:, 1],
        "label_arc": label_arc, "label_dist": label_dist, "label_valid": label_valid,
        "n_cand": [len(c[0]) for c in cands],
        "label_in_cands": [la in set(c[0].tolist()) for la, c in zip(label_arc, cands)],
    })
    return df, point_info

"""Candidate-level and observation/route-level metrics required by the guidelines."""
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             log_loss, confusion_matrix)

from .io_utils import load_trip


def candidate_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def top1_predictions(cand, prob_col="prob"):
    """Highest-scoring candidate per GPS point -> predicted arc."""
    best = cand.loc[cand.groupby(["trip", "obs"])[prob_col].idxmax(), ["trip", "obs", "arc_id"]]
    return best.rename(columns={"arc_id": "pred_arc"})


@lru_cache(maxsize=256)
def _arcs(partition, trip_id):
    return load_trip(partition, trip_id).arcs


def route_continuity(pred_df):
    """Share of consecutive predicted arcs that are identical or directly connected
    (end node of the first = start node of the second)."""
    ok = tot = 0
    for (part, tid), g in pred_df[pred_df.pred_arc >= 0].sort_values("obs").groupby(["partition", "trip"]):
        arcs = _arcs(part, tid)
        a, b = g.pred_arc.values[:-1], g.pred_arc.values[1:]
        good = (a == b) | (arcs[a, 1] == arcs[b, 0])
        ok += good.sum(); tot += len(good)
    return ok / max(tot, 1)


def observation_metrics(points, pred):
    """points: per-point table (has label_arc, label_valid). pred: trip, obs, pred_arc.
    Points without any candidate count as wrong (pred_arc = -1)."""
    m = points.merge(pred, on=["trip", "obs"], how="left")
    m["pred_arc"] = m["pred_arc"].fillna(-1).astype(int)
    ev = m[m.label_valid]
    return {
        "top1_accuracy": float((ev.pred_arc == ev.label_arc).mean()),
        "candidate_miss_rate": float(1 - ev.label_in_cands.mean()),
        "route_continuity": float(route_continuity(m)),
        "n_eval_points": int(len(ev)),
    }, m

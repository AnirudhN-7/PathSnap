"""Step 6: classifier-only vs classifier + HMM/Viterbi on the student-test trips.
Outputs (outputs/hmm/): hmm_vs_classifier_<model>.csv (overall, one row per beta),
        per_trip_<model>.csv, predictions_<model>.parquet
Usage:  python -m scripts.run_hmm [--model hist_gradient_boosting_tuned] [--betas 0.5,1,2,3,5,10,30,60]"""
import argparse, time
import joblib
import numpy as np
import pandas as pd

from src.config import PROC_DIR, OUT_DIR, MODEL_DIR, HMM_BETA_M, FINAL_MODEL
from src.features import FEATURES
from src.io_utils import load_trip
from src.hmm import RoadGraph, viterbi_trip
from src.metrics import top1_predictions, observation_metrics


def route_set_scores(pred_arcs, route):
    """Sequence-level view: overlap between the set of predicted arcs and the reference route."""
    p, r = set(pred_arcs[pred_arcs >= 0].tolist()), set(route.tolist())
    tp = len(p & r)
    prec, rec = tp / max(len(p), 1), tp / max(len(r), 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=FINAL_MODEL)
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--betas", default=str(HMM_BETA_M))
    args = ap.parse_args()
    out = OUT_DIR / "hmm"; out.mkdir(parents=True, exist_ok=True)

    cand = pd.read_parquet(PROC_DIR / f"{args.partition}_candidates.parquet")
    pts = pd.read_parquet(PROC_DIR / f"{args.partition}_points.parquet")
    model = joblib.load(MODEL_DIR / f"{args.model}.joblib")
    cand["prob"] = model.predict_proba(cand[FEATURES].values)[:, 1]
    clf_pred = top1_predictions(cand)

    trips = {tid: load_trip(args.partition, tid) for tid in sorted(cand.trip.unique())}
    graphs = {tid: RoadGraph(tr.node_xy, tr.arcs) for tid, tr in trips.items()}
    groups = {tid: {o: (g.arc_id.values, g.proj_t.values, g.prob.values) for o, g in df.groupby("obs")}
              for tid, df in cand.groupby("trip")}

    overall, per_trip, best = [], [], None
    for beta in [float(b) for b in args.betas.split(",")]:
        rows, t0 = [], time.time()
        for tid, tr in trips.items():
            p = viterbi_trip(graphs[tid], tr.xy, tr.track[:, 2], groups[tid], beta=beta)
            rows.append(pd.DataFrame({"trip": tid, "obs": np.arange(len(p)), "pred_arc": p}))
        hmm_pred = pd.concat(rows)
        hmm_pred = hmm_pred[hmm_pred.pred_arc >= 0]
        decode_s = time.time() - t0
        m_clf, merged_clf = observation_metrics(pts, clf_pred)
        m_hmm, merged_hmm = observation_metrics(pts, hmm_pred)
        overall.append({"model": args.model, "beta": beta, "decode_time_s": decode_s,
                        **{f"clf_{k}": v for k, v in m_clf.items()}, **{f"hmm_{k}": v for k, v in m_hmm.items()}})
        print(f"beta={beta}: top1 {m_clf['top1_accuracy']:.4f} -> {m_hmm['top1_accuracy']:.4f} | "
              f"continuity {m_clf['route_continuity']:.4f} -> {m_hmm['route_continuity']:.4f} ({decode_s:.0f}s)")
        if best is None or m_hmm["top1_accuracy"] > best[0]:
            best = (m_hmm["top1_accuracy"], beta, merged_clf, merged_hmm)

    _, beta, mc, mh = best
    for tid, tr in trips.items():
        c, h = mc[mc.trip == tid], mh[mh.trip == tid]
        ev_c, ev_h = c[c.label_valid], h[h.label_valid]
        rc, rh = route_set_scores(c.pred_arc.values, tr.route), route_set_scores(h.pred_arc.values, tr.route)
        per_trip.append({"trip": tid, "n_pts": len(c), "beta": beta,
                         "clf_top1": (ev_c.pred_arc == ev_c.label_arc).mean(),
                         "hmm_top1": (ev_h.pred_arc == ev_h.label_arc).mean(),
                         "clf_route_f1": rc[2], "hmm_route_precision": rh[0],
                         "hmm_route_recall": rh[1], "hmm_route_f1": rh[2]})
    pd.DataFrame(overall).to_csv(out / f"hmm_vs_classifier_{args.model}.csv", index=False)
    pt = pd.DataFrame(per_trip); pt.to_csv(out / f"per_trip_{args.model}.csv", index=False)
    mh[["trip", "obs", "pred_arc"]].merge(mc[["trip", "obs", "pred_arc"]].rename(columns={"pred_arc": "clf_arc"}),
                                          on=["trip", "obs"]).to_parquet(out / f"predictions_{args.model}.parquet", index=False)
    print(pt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Step 7: HMM figures + Folium maps of predicted routes (reads outputs/hmm/, written by run_hmm).
 - hmm_vs_classifier_per_trip.png : chosen model, classifier-only vs +HMM per student-test trip
 - all_models_classifier_vs_hmm.png : every model through the same decoder
 - maps/student_test_<trip>_<model>.html : trips where the HMM helped most and least
Usage:  python -m scripts.hmm_plots [--model hist_gradient_boosting]"""
import argparse
import numpy as np
import pandas as pd

from src.config import OUT_DIR, FINAL_MODEL
from src.io_utils import load_trip
from src.plotting import plt, save, SERIES
from src.visualize import trip_map


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=FINAL_MODEL)
    args = ap.parse_args()
    fig_dir = OUT_DIR / "figures"; (OUT_DIR / "maps").mkdir(parents=True, exist_ok=True)
    # 7. classifier-only vs HMM, per student-test trip
    pt_file = OUT_DIR / "hmm" / f"per_trip_{args.model}.csv"
    if pt_file.exists():
        pt = pd.read_csv(pt_file, dtype={"trip": str}); yy = np.arange(len(pt)); h = 0.38
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.barh(yy - h / 2, pt.clf_top1, height=h - 0.04, color=SERIES[0], label="Classifier only")
        ax.barh(yy + h / 2, pt.hmm_top1, height=h - 0.04, color=SERIES[1], label="Classifier + HMM/Viterbi")
        ax.set_yticks(yy, pt.trip); ax.set_xlim(0.6, 1); ax.invert_yaxis()
        ax.set(title=f"Top-1 accuracy per student-test trip ({args.model})", xlabel="top-1 accuracy")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); save(fig, fig_dir / "hmm_vs_classifier_per_trip.png")

        # Folium map for the trip where the HMM helped most, and the one where it helped least
        preds = pd.read_parquet(OUT_DIR / "hmm" / f"predictions_{args.model}.parquet")
        gain = (pt.hmm_top1 - pt.clf_top1)
        for tid in {pt.trip[gain.idxmax()], pt.trip[gain.idxmin()]}:
            trip = load_trip("student_test", tid)
            p = preds[preds.trip == tid].sort_values("obs")
            full = np.full(len(trip.track), -1); full[p.obs.values] = p.pred_arc.values
            fullc = np.full(len(trip.track), -1); fullc[p.obs.values] = p.clf_arc.values
            trip_map(trip, clf_arcs=fullc, hmm_arcs=full).save(str(OUT_DIR / "maps" / f"student_test_{tid}_{args.model}.html"))
    # 8. every model before vs after HMM (the metric used for final selection)
    rows = []
    for f in sorted((OUT_DIR / "hmm").glob("hmm_vs_classifier_*.csv")):
        d = pd.read_csv(f)
        b = d.loc[(d.beta - 2.0).abs().idxmin()]          # same beta for every model
        rows.append({"model": b.model, "beta": b.beta, "classifier_top1": b.clf_top1_accuracy,
                     "hmm_top1": b.hmm_top1_accuracy, "classifier_continuity": b.clf_route_continuity,
                     "hmm_continuity": b.hmm_route_continuity})
    if rows:
        allm = pd.DataFrame(rows).sort_values("hmm_top1")
        allm.to_csv(OUT_DIR / "hmm" / "all_models_classifier_vs_hmm.csv", index=False)
        yy = np.arange(len(allm)); h = 0.38
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(yy + h / 2, allm.classifier_top1, height=h - 0.04, color=SERIES[0], label="Classifier only")
        ax.barh(yy - h / 2, allm.hmm_top1, height=h - 0.04, color=SERIES[1], label="Classifier + HMM/Viterbi")
        ax.set_yticks(yy, allm.model); ax.set_xlim(0.75, 1)
        ax.set(title=f"Student-test top-1 accuracy, all models (beta = {allm.beta.iloc[0]:g} m)", xlabel="top-1 accuracy")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        save(fig, fig_dir / "all_models_classifier_vs_hmm.png")
    print("HMM figures and maps written")


if __name__ == "__main__":
    main()

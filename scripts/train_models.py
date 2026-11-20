"""Step 3: train + compare the five models.
 - 5-fold GroupKFold over the 80 training trips (a trip never spans two folds)
 - refit on all usable training trips, evaluate once on the 10 student-test trips
Outputs (outputs/experiments/): model_comparison.csv, cv_folds.csv, confusion_matrices.json,
         mlp_loss_curve.csv, run_log.json; models/<name>.joblib; outputs/predictions/<name>_student_test.parquet
Usage:  python -m scripts.train_models [--models a,b] [--max-train-rows 400000] [--folds 5]"""
import argparse, json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import PROC_DIR, OUT_DIR, MODEL_DIR, SEED, BAD_TRIP_FRAC
from src.features import FEATURES
from src.metrics import candidate_metrics, top1_predictions, observation_metrics
from src.models import MODEL_SPECS


def load(part):
    cand = pd.read_parquet(PROC_DIR / f"{part}_candidates.parquet")
    pts = pd.read_parquet(PROC_DIR / f"{part}_points.parquet")
    return cand, pts


def usable_train(cand, pts):
    bad = pts.groupby("trip").label_valid.mean()
    bad = bad[1 - bad > BAD_TRIP_FRAC].index
    cand = cand[~cand.trip.isin(bad) & cand.label_valid].reset_index(drop=True)
    pts = pts[~pts.trip.isin(bad)].reset_index(drop=True)
    return cand, pts, list(bad)


def subsample(df, n, seed):
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


def fit(name, df, max_rows, seed):
    sub = subsample(df, max_rows, seed)
    model = MODEL_SPECS[name]["build"]()
    t0 = time.time()
    model.fit(sub[FEATURES].values, sub["y"].values)
    return model, time.time() - t0, len(sub)


def evaluate(model, cand, pts):
    t0 = time.time()
    prob = model.predict_proba(cand[FEATURES].values)[:, 1]
    infer_s = time.time() - t0
    c = cand[["trip", "partition", "obs", "arc_id", "y", "label_valid"]].copy()
    c["prob"] = prob
    ev = c[c.label_valid]
    cm = candidate_metrics(ev.y.values, ev.prob.values)
    om, _ = observation_metrics(pts, top1_predictions(c))
    return cm, om, infer_s, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_SPECS))
    ap.add_argument("--max-train-rows", type=int, default=400_000)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    exp = OUT_DIR / "experiments"; pred_dir = OUT_DIR / "predictions"
    for d in (exp, pred_dir, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    tr_c, tr_p = load("train")
    tr_c, tr_p, excluded = usable_train(tr_c, tr_p)
    te_c, te_p = load("student_test")
    print(f"train rows {len(tr_c)} from {tr_c.trip.nunique()} trips (excluded {excluded}); "
          f"student-test rows {len(te_c)}")

    gkf = GroupKFold(n_splits=args.folds)
    trips = tr_c.trip.values
    fold_rows, summary, confusions = [], [], {}
    for name in args.models.split(","):
        print(f"\n=== {name}")
        for k, (i_tr, i_va) in enumerate(gkf.split(tr_c, groups=trips)):
            va_trips = set(trips[i_va])
            model, fit_s, n_fit = fit(name, tr_c.iloc[i_tr], args.max_train_rows, SEED + k)
            cm, om, inf_s, _ = evaluate(model, tr_c.iloc[i_va], tr_p[tr_p.trip.isin(va_trips)])
            # same metrics on the fold's own training trips -> train-vs-validation gap (over/underfitting evidence)
            tr_trips = set(trips[i_tr])
            cm_tr, om_tr, _, _ = evaluate(model, tr_c.iloc[i_tr], tr_p[tr_p.trip.isin(tr_trips)])
            fold_rows.append({"model": name, "fold": k, "fit_s": fit_s, **{x: cm[x] for x in cm if x != "confusion"}, **om,
                              "train_f1_macro": cm_tr["f1_macro"], "train_log_loss": cm_tr["log_loss"],
                              "train_top1_accuracy": om_tr["top1_accuracy"]})
            print(f"  fold {k}: val F1={cm['f1_macro']:.4f} (train {cm_tr['f1_macro']:.4f}) "
                  f"logloss={cm['log_loss']:.4f} top1={om['top1_accuracy']:.4f} ({fit_s:.1f}s)")

        model, fit_s, n_fit = fit(name, tr_c, args.max_train_rows, SEED)
        cm, om, inf_s, c = evaluate(model, te_c, te_p)
        confusions[name] = cm.pop("confusion")
        joblib.dump(model, MODEL_DIR / f"{name}.joblib", compress=3)
        c.to_parquet(pred_dir / f"{name}_student_test.parquet", index=False)
        if name == "mlp":
            mlp = model[-1]
            pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1), "train_loss": mlp.loss_curve_,
                          "val_accuracy": mlp.validation_scores_}).to_csv(exp / "mlp_loss_curve.csv", index=False)
        cv = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        summary.append({
            "model": name, "family": MODEL_SPECS[name]["family"], "config": MODEL_SPECS[name]["config"],
            "cv_f1_macro_mean": cv.f1_macro.mean(), "cv_f1_macro_std": cv.f1_macro.std(),
            "cv_top1_mean": cv.top1_accuracy.mean(), "cv_log_loss_mean": cv.log_loss.mean(),
            "cv_train_f1_macro_mean": cv.train_f1_macro.mean(), "cv_train_log_loss_mean": cv.train_log_loss.mean(),
            "cv_train_top1_mean": cv.train_top1_accuracy.mean(),
            "test_accuracy": cm["accuracy"], "test_precision_macro": cm["precision_macro"],
            "test_recall_macro": cm["recall_macro"], "test_f1_macro": cm["f1_macro"],
            "test_log_loss": cm["log_loss"], "test_top1_accuracy": om["top1_accuracy"],
            "test_route_continuity": om["route_continuity"], "test_candidate_miss_rate": om["candidate_miss_rate"],
            "train_rows_used": n_fit, "fit_time_s": fit_s, "inference_time_s": inf_s,
            "inference_ms_per_1k_rows": 1000 * inf_s / len(te_c) * 1000,
        })
        print(f"  student-test: F1={cm['f1_macro']:.4f} logloss={cm['log_loss']:.4f} "
              f"top1={om['top1_accuracy']:.4f} continuity={om['route_continuity']:.4f}")

    # merge with earlier runs so models can be trained on different laptops / at different times
    out = exp / "model_comparison.csv"; folds_out = exp / "cv_folds.csv"
    new = pd.DataFrame(summary); new_f = pd.DataFrame(fold_rows)
    if out.exists():
        new = pd.concat([pd.read_csv(out).query("model not in @new.model"), new])
        new_f = pd.concat([pd.read_csv(folds_out).query("model not in @new_f.model"), new_f])
    new.to_csv(out, index=False); new_f.to_csv(folds_out, index=False)
    cfile = exp / "confusion_matrices.json"
    old = json.loads(cfile.read_text()) if cfile.exists() else {}
    cfile.write_text(json.dumps({**old, **confusions}, indent=2))
    (exp / "run_log.json").write_text(json.dumps({"args": vars(args), "seed": SEED, "features": FEATURES,
                                                  "excluded_trips": excluded,
                                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print("\n", new[["model", "cv_f1_macro_mean", "test_f1_macro", "test_log_loss",
                    "test_top1_accuracy", "test_route_continuity", "inference_time_s"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

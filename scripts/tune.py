"""Step 5: Bayesian hyperparameter optimisation (Optuna TPE) of the chosen model, HistGradientBoosting.

Objective : maximise mean observation-level Top-1 accuracy over a 3-fold GroupKFold split of the training
            trips (200k sampled rows per fold, to keep 30 trials affordable on a laptop).
            Student-test and evaluator-test data are never seen by the search.
Compare   : untuned vs tuned with the SAME 5-fold GroupKFold split and 400k-row budget as train_models.py,
            and the same metrics: CV macro F1, student-test top-1, student-test macro F1, log loss, inference time.
Outputs (outputs/tuning/): search_space.json, optuna_trials.csv, best_params.json, tuned_vs_untuned.csv,
         optimisation_history.png, tuned_vs_untuned.png;  models/hist_gradient_boosting_tuned.joblib
Usage:  python -m scripts.tune [--trials 30]"""
import argparse, json, time
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.config import OUT_DIR, MODEL_DIR, SEED
from src.features import FEATURES
from src.models import MODEL_SPECS
from src.plotting import plt, save, SERIES, INK2
from scripts.train_models import load, usable_train, subsample, evaluate

# search space: name -> (type, low, high, log-scale?)  saved to JSON so the report can cite it exactly
SEARCH_SPACE = {
    "learning_rate":     ("float", 0.02, 0.30, True),   # step size; smaller needs more iterations
    "max_iter":          ("int",   100, 600, False),    # number of boosting rounds (upper bound, early stopping)
    "max_leaf_nodes":    ("int",   15, 127, True),      # tree complexity
    "min_samples_leaf":  ("int",   20, 400, True),      # smoothing / regularisation of leaves
    "l2_regularization": ("float", 1e-4, 10.0, True),   # penalty on leaf values
    "max_features":      ("float", 0.5, 1.0, False),    # feature subsampling per split (decorrelates trees)
}


def suggest(trial):
    p = {}
    for name, (kind, lo, hi, log) in SEARCH_SPACE.items():
        if kind == "int":
            p[name] = trial.suggest_int(name, lo, hi, log=log) if log else trial.suggest_int(name, lo, hi, step=50)
        else:
            p[name] = trial.suggest_float(name, lo, hi, log=log)
    return p


def build(p):
    return HistGradientBoostingClassifier(random_state=SEED, early_stopping=True, validation_fraction=0.1, **p)


def cross_validate(make_model, tr_c, tr_p, splits, max_rows):
    f1s, top1s, losses = [], [], []
    for k, (i_tr, i_va) in enumerate(splits):
        sub = subsample(tr_c.iloc[i_tr], max_rows, SEED + k)        # same sample as train_models.py
        m = make_model().fit(sub[FEATURES].values, sub.y.values)
        va = tr_c.iloc[i_va]
        cm, om, _, _ = evaluate(m, va, tr_p[tr_p.trip.isin(set(va.trip))])
        f1s.append(cm["f1_macro"]); top1s.append(om["top1_accuracy"]); losses.append(cm["log_loss"])
    return np.mean(f1s), np.std(f1s, ddof=1), np.mean(top1s), np.mean(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--search-folds", type=int, default=3)
    ap.add_argument("--search-rows", type=int, default=200_000)
    ap.add_argument("--folds", type=int, default=5)               # final comparison = train_models.py setup
    ap.add_argument("--max-train-rows", type=int, default=400_000)
    args = ap.parse_args()
    out = OUT_DIR / "tuning"; out.mkdir(parents=True, exist_ok=True)
    (out / "search_space.json").write_text(json.dumps(
        {k: {"type": t, "low": lo, "high": hi, "log_scale": lg} for k, (t, lo, hi, lg) in SEARCH_SPACE.items()}
        | {"_sampler": "TPE (Optuna), seed 42", "_objective": f"mean observation top-1 accuracy, {args.search_folds}-fold GroupKFold over training trips, {args.search_rows:,} rows per fold",
           "_fixed": "early_stopping=True, validation_fraction=0.1, random_state=42"}, indent=2))

    tr_c, tr_p = load("train"); tr_c, tr_p, _ = usable_train(tr_c, tr_p)
    te_c, te_p = load("student_test")
    search_splits = list(GroupKFold(n_splits=args.search_folds).split(tr_c, groups=tr_c.trip.values))
    splits = list(GroupKFold(n_splits=args.folds).split(tr_c, groups=tr_c.trip.values))

    def objective(trial):
        p = suggest(trial)
        f1, f1_sd, top1, loss = cross_validate(lambda: build(p), tr_c, tr_p, search_splits, args.search_rows)
        trial.set_user_attr("cv_f1_macro", f1); trial.set_user_attr("cv_log_loss", loss)
        print(f"trial {trial.number:2d}: cv top1={top1:.4f} f1={f1:.4f} logloss={loss:.4f}", flush=True)
        return top1

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.trials)
    tune_s = time.time() - t0
    trials = study.trials_dataframe(); trials.to_csv(out / "optuna_trials.csv", index=False)
    (out / "best_params.json").write_text(json.dumps({"best_cv_top1": study.best_value, "best_trial": study.best_trial.number,
                                                      "params": study.best_params, "n_trials": args.trials,
                                                      "tuning_time_s": round(tune_s, 1)}, indent=2))

    rows = []
    variants = [("untuned", lambda: MODEL_SPECS["hist_gradient_boosting"]["build"]()),
                ("tuned", lambda: build(study.best_params))]
    for label, make in variants:
        f1, f1_sd, cv_top1, cv_loss = cross_validate(make, tr_c, tr_p, splits, args.max_train_rows)
        sub = subsample(tr_c, args.max_train_rows, SEED)
        t1 = time.time(); model = make().fit(sub[FEATURES].values, sub.y.values); fit_s = time.time() - t1
        cm, om, inf_s, _ = evaluate(model, te_c, te_p)
        rows.append({"model": label, "cv_f1_macro_mean": f1, "cv_f1_macro_std": f1_sd, "cv_top1_mean": cv_top1,
                     "test_top1_accuracy": om["top1_accuracy"], "test_f1_macro": cm["f1_macro"],
                     "test_log_loss": cm["log_loss"], "test_route_continuity": om["route_continuity"],
                     "fit_time_s": fit_s, "inference_time_s": inf_s,
                     "n_iter_used": int(model.n_iter_)})
        if label == "tuned":
            joblib.dump(model, MODEL_DIR / "hist_gradient_boosting_tuned.joblib", compress=3)
    res = pd.DataFrame(rows); res.to_csv(out / "tuned_vs_untuned.csv", index=False)

    # optimisation history
    v = trials.value.values
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(trials.number, v, s=16, color=SERIES[0], label="Trial objective")
    ax.plot(trials.number, np.maximum.accumulate(v), color=SERIES[1], label="Best so far")
    ax.set(title="Bayesian optimisation history (Optuna TPE)", xlabel="trial", ylabel="CV top-1 accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); save(fig, out / "optimisation_history.png")

    # tuned vs untuned
    metrics = [("cv_f1_macro_mean", "CV macro F1"), ("test_top1_accuracy", "Test top-1"),
               ("test_f1_macro", "Test macro F1")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [3, 1, 1]})
    yy = np.arange(len(metrics)); h = 0.38
    for i, (lab, color) in enumerate([("untuned", SERIES[0]), ("tuned", SERIES[1])]):
        r = res[res.model == lab].iloc[0]
        axes[0].barh(yy + (h / 2 if i == 0 else -h / 2), [r[m] for m, _ in metrics], height=h - 0.04, color=color, label=lab)
        axes[1].bar(i, r.test_log_loss, color=color, width=0.6); axes[2].bar(i, r.inference_time_s, color=color, width=0.6)
    axes[0].set_yticks(yy, [n for _, n in metrics]); axes[0].set_xlim(0.8, 1); axes[0].invert_yaxis()
    axes[0].set_title("Scores (higher is better)"); axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    for ax, t in [(axes[1], "Test log loss"), (axes[2], "Inference time (s)")]:
        ax.set_xticks([0, 1], ["untuned", "tuned"]); ax.set_title(t)
    fig.tight_layout(); save(fig, out / "tuned_vs_untuned.png")

    print(f"\nbest params (trial {study.best_trial.number}): {json.dumps(study.best_params)}")
    print(f"tuning time {tune_s:.0f}s for {args.trials} trials")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

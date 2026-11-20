"""Step 4: model-comparison figures from the saved experiment results (never from hand-typed numbers).
Usage:  python -m scripts.make_plots"""
import json
import numpy as np
import pandas as pd

from src.config import OUT_DIR
from src.plotting import plt, save, SERIES, INK2


def main():
    exp = OUT_DIR / "experiments"; fig_dir = OUT_DIR / "figures"
    res = pd.read_csv(exp / "model_comparison.csv"); folds = pd.read_csv(exp / "cv_folds.csv")
    order = res.sort_values("test_top1_accuracy").model.tolist()
    res = res.set_index("model").loc[order].reset_index()
    y = np.arange(len(res))

    # 1. CV macro F1 (mean ± std over 5 GroupKFold folds) + fold dots
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(y, res.cv_f1_macro_mean, height=0.55, color=SERIES[0])
    ax.errorbar(res.cv_f1_macro_mean, y, xerr=res.cv_f1_macro_std, fmt="none", ecolor=INK2, capsize=3, lw=1)
    for i, m in enumerate(res.model):
        f = folds[folds.model == m].f1_macro
        ax.scatter(f, np.full(len(f), i), s=10, color="#0b0b0b", zorder=3)
    ax.set_yticks(y, res.model); ax.set(title="Cross-validation macro F1 (5-fold GroupKFold, train)", xlabel="macro F1")
    ax.set_xlim(max(0, res.cv_f1_macro_mean.min() - 0.1), 1)
    save(fig, fig_dir / "cv_macro_f1.png")

    # 2. student-test top-1 accuracy and macro F1
    fig, ax = plt.subplots(figsize=(7, 3.8)); h = 0.38
    ax.barh(y + h / 2, res.test_top1_accuracy, height=h - 0.04, color=SERIES[0], label="Top-1 accuracy (per GPS point)")
    ax.barh(y - h / 2, res.test_f1_macro, height=h - 0.04, color=SERIES[1], label="Macro F1 (per candidate)")
    ax.set_yticks(y, res.model); ax.set_xlim(0.6, 1)
    ax.set(title="Student-test performance (classifier only)", xlabel="score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); save(fig, fig_dir / "student_test_top1_f1.png")

    # 3. log loss
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(y, res.test_log_loss, height=0.55, color=SERIES[0]); ax.set_xscale("log")
    for i, v in enumerate(res.test_log_loss): ax.text(v * 1.05, i, f"{v:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, res.model); ax.set(title="Student-test log loss (lower is better, log scale)", xlabel="log loss")
    save(fig, fig_dir / "student_test_log_loss.png")

    # 4. inference time
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(y, res.inference_time_s, height=0.55, color=SERIES[0])
    ax.set_yticks(y, res.model); ax.set(title="Inference time on all student-test candidates", xlabel="seconds")
    save(fig, fig_dir / "inference_time.png")

    # 5. confusion matrices (optional diagnostic)
    cms = json.loads((exp / "confusion_matrices.json").read_text())
    fig, axes = plt.subplots(1, len(order), figsize=(3 * len(order), 3))
    for ax, m in zip(np.atleast_1d(axes), order):
        c = np.array(cms[m]); ax.imshow(c / c.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(c):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                    color="#fcfcfb" if c[i, j] / c[i].sum() > 0.5 else "#0b0b0b")
        ax.set_xticks([0, 1], ["not route", "route"]); ax.set_yticks([0, 1], ["not route", "route"])
        ax.set_title(m, fontsize=9); ax.grid(False); ax.set_xlabel("predicted")
    np.atleast_1d(axes)[0].set_ylabel("actual"); fig.tight_layout()
    save(fig, fig_dir / "confusion_matrices.png")

    # 6. MLP training history (only model with epoch-wise history)
    if (exp / "mlp_loss_curve.csv").exists():
        lc = pd.read_csv(exp / "mlp_loss_curve.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].plot(lc.epoch, lc.train_loss, color=SERIES[0]); ax[0].set(title="MLP training loss", xlabel="epoch", ylabel="log loss")
        ax[1].plot(lc.epoch, lc.val_accuracy, color=SERIES[0]); ax[1].set(title="MLP validation accuracy (early-stopping split)", xlabel="epoch", ylabel="accuracy")
        fig.tight_layout(); save(fig, fig_dir / "mlp_training_curve.png")

    # 7. train vs validation macro F1 per model (over/underfitting evidence for every model)
    if "train_f1_macro" in folds:
        g = folds.groupby("model")[["train_f1_macro", "f1_macro"]].mean().loc[order]
        fig, ax = plt.subplots(figsize=(7, 3.8)); h = 0.38
        ax.barh(y + h / 2, g.train_f1_macro, height=h - 0.04, color=SERIES[0], label="Training trips (in-fold)")
        ax.barh(y - h / 2, g.f1_macro, height=h - 0.04, color=SERIES[1], label="Validation trips (held-out fold)")
        for i, (a, b) in enumerate(zip(g.train_f1_macro, g.f1_macro)):
            ax.text(max(a, b) + 0.005, i, f"gap {a - b:+.3f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, order); ax.set_xlim(0.6, 1.05)
        ax.set(title="Train vs validation macro F1 (mean over 5 folds)", xlabel="macro F1")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        save(fig, fig_dir / "train_vs_validation_f1.png")

    print("figures written to", fig_dir)


if __name__ == "__main__":
    main()

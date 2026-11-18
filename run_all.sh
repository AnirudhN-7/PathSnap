#!/usr/bin/env bash
# Reproduces every offline result in outputs/ (~25-30 min on a laptop). The API/UI are started separately (README).
set -euo pipefail
cd "$(dirname "$0")"
python -m scripts.eda                       # 1  EDA                     -> outputs/eda
python -m scripts.build_dataset             # 2  labels + candidates      -> data/processed, outputs/data_quality
python -m scripts.train_models              # 3  5 models, CV + test      -> outputs/experiments, models/
python -m scripts.make_plots                # 4  comparison figures       -> outputs/figures
python -m scripts.tune --trials 30          # 5  Bayesian optimisation    -> outputs/tuning, models/*_tuned.joblib
python -m scripts.run_hmm --betas 0.5,1,2,3,5,10,30,60      # 6  HMM beta sweep, final model -> outputs/hmm
python -m scripts.run_hmm --model hist_gradient_boosting_tuned --betas 0.5,1,2,3,5,10,30,60   # tuned model
for m in random_forest mlp logistic_regression gaussian_nb; do
  python -m scripts.run_hmm --model "$m" --betas 2          #    every other model through the same decoder
done
python -m scripts.hmm_plots                 # 7  HMM figures + Folium maps -> outputs/figures, outputs/maps
python -m scripts.report_tables             # 8  report tables            -> report/results_tables.md
echo "Done."

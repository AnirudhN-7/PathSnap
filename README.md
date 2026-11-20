# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon. Noisy 1 Hz GPS tracks are matched to a road network:
1. A classifier scores the nearby road segments for every GPS point.
2. An HMM/Viterbi decoder turns those scores into a connected route.
3. The system is deployed as a FastAPI backend (laptop 1) plus a Streamlit evaluator UI (laptop 2).

```
 raw trip files (.track .nodes .arcs .route)
   │ scripts/eda.py ............ exploration, data-quality log, plots, Folium map
   │ scripts/build_dataset.py .. labels (route alignment) → candidates ≤50 m → 18 features
   ▼
 (GPS point, candidate arc) table
   │ scripts/train_models.py ... 5 models · 5-fold GroupKFold CV · student-test evaluation
   │ scripts/make_plots.py ..... comparison figures
   │ scripts/tune.py ........... Bayesian optimisation (Optuna TPE) of HistGradientBoosting
   ▼
 P(candidate is the true road segment)
   │ scripts/run_hmm.py ........ HMM/Viterbi, classifier-only vs +HMM, beta sweep
   │ scripts/hmm_plots.py ...... HMM figures + Folium maps of predicted routes
   ▼
 api/main.py (laptop 1)  ◄── HTTP ──►  ui/app.py (laptop 2)
 
```

## 1. Installation (macOS)

```bash
git clone https://github.com/<owner>/pathsnap.git && cd pathsnap
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # laptop 1 / anyone running experiments
pip install -r ui/requirements.txt         # laptop 2 (UI) - can be a separate venv
```

**Data is not committed.** Copy the provided folders so the layout is:

```
data/raw/train/00000001/00000001.{track,nodes,arcs,route}
data/raw/student_test/00000000/...
data/evaluation/<trip_id>/...      # ONLY on laptop 1, ONLY at evaluation time (the 10 reserved trips)
```

No training, tuning or threshold script ever reads `data/evaluation`.

## 2. Reproduce all results

```bash
./run_all.sh        # ~25-30 min; writes everything in outputs/ and report/results_tables.md
```

| # | Command | Produces |
|---|---|---|
| 1 | `python -m scripts.eda` | `outputs/eda/`: inventory, per-trip stats, partition summary, data-quality issues, plots, Folium map |
| 2 | `python -m scripts.build_dataset` | `data/processed/*.parquet`, `outputs/data_quality/trip_label_quality.csv` |
| 3 | `python -m scripts.train_models` | `outputs/experiments/` (`model_comparison.csv`, `cv_folds.csv`, `confusion_matrices.json`, `mlp_loss_curve.csv`, `run_log.json`), `models/*.joblib`, `outputs/predictions/` |
| 4 | `python -m scripts.make_plots` | `outputs/figures/`: CV F1, test top-1/F1, log loss, inference time, confusion matrices, MLP curve, train-vs-validation |
| 5 | `python -m scripts.tune --trials 30` | `outputs/tuning/` (search space, all trials, best params, tuned vs untuned, plots), `models/hist_gradient_boosting_tuned.joblib` |
| 6 | `python -m scripts.run_hmm --betas 0.5,1,2,3,5,10,30,60` | `outputs/hmm/`: beta sweep, per-trip results, predictions (final model); repeat with `--model hist_gradient_boosting_tuned` |
| 6b | `python -m scripts.run_hmm --model <name> --betas 2` | same decoder for any other model |
| 7 | `python -m scripts.hmm_plots` | `outputs/figures/*hmm*.png`, `outputs/maps/*.html` |
| 8 | `python -m scripts.report_tables` | `report/results_tables.md`: every report table, read from the CSVs |

All randomness uses `SEED = 42`. All settings live in `src/config.py`.

## 3. Deployment (two laptops)

Both laptops must be on the **same network**. A phone hotspot works if college Wi-Fi blocks laptop-to-laptop traffic.

**Laptop 1: model + API**

```bash
cd pathsnap && source .venv/bin/activate
# put the 10 reserved trips in data/evaluation/<trip_id>/  (or point elsewhere:)
# export PATHSNAP_EVAL_DIR=/path/to/evaluation_trips
uvicorn api.main:app --host 0.0.0.0 --port 8000
ipconfig getifaddr en0          # laptop 1's IP address, e.g. 192.168.1.23
```

- The terminal shows one log block per request: trip ID, stage timings, accuracy. Keep it visible for the evaluators.
- On first start, macOS asks whether to accept incoming connections: click **Allow**.
- To rehearse before the evaluation, use `export PATHSNAP_EVAL_DIR=$PWD/data/raw/student_test`.

**Laptop 2: UI**

```bash
cd pathsnap && source .venv/bin/activate
export PATHSNAP_API_URL=http://192.168.1.23:8000      # laptop 1's IP
streamlit run ui/app.py
```

The browser opens at `http://localhost:8501`. The sidebar shows whether the API is reachable. The evaluator picks one of the 10 trips and clicks **Run map-matching**.

**API endpoints** (interactive docs at `http://<laptop-1-ip>:8000/docs`)

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | model name, HMM beta, evaluation folder, number of trips |
| GET | `/trips` | none | the evaluation trips: ID, number of GPS points, duration |
| POST | `/predict` | `{"trip_id": "00000077"}` | predicted arc per GPS point, map geometry, server-side metrics, timings |

**Integrity guarantees**
- The prediction step loads the trip with `with_route=False`, so the reference route is never read. The pipeline asserts this.
- The route is only read afterwards, to compute accuracy on laptop 1.
- Responses contain accuracy numbers but never the reference route.
- The UI only talks to the API; it has no access to the dataset or the model.

**Check every trip:** `python -m scripts.api_smoke_test --url http://<laptop-1-ip>:8000` writes `outputs/api_logs/api_smoke_test.csv`.

## 4. Repository layout

```
src/config.py       every constant: paths, radius, K, label thresholds, HMM beta/gap, final model
src/io_utils.py     trip loader, lon/lat → metres projection
src/geometry.py     point-to-segment projection, densified KD-tree arc index (candidate search)
src/labels.py       monotone DP alignment of GPS points to the .route arc sequence
src/features.py     candidate generation + 18 features
src/metrics.py      F1/precision/recall/log loss, top-1, candidate miss rate, route continuity
src/models.py       the 5 model definitions, initial configurations, loss functions
src/hmm.py          HMM + Viterbi decoder
src/pipeline.py     predict_trip() / evaluate_trip(), used by the API
src/visualize.py    Folium maps;  src/plotting.py  shared figure style
scripts/            runnable steps (table above) + api_smoke_test.py + report_tables.py
api/main.py         FastAPI backend (laptop 1)
ui/app.py           Streamlit evaluator UI (laptop 2)
outputs/            all results (CSV/JSON/PNG/HTML), committed so the report is verifiable
models/             trained models (joblib)
report/             results_tables.md (generated), screenshots/
```

## 5. Key design decisions

| Decision | Evidence |
|---|---|
| Per-trip local projection to metres | Trips span the globe (lon −100…137, lat −34…62), but each is < 80 km wide |
| Labels from monotone DP alignment | `.route` has no timestamps and fewer rows than `.track`; routes can loop |
| Trips 00000033 and 00000054 excluded from training | More than 20% of their points are > 50 m from the reference route (`trip_label_quality.csv`) |
| Candidates within 50 m, at most 15 | Candidate miss rate about 0.1% on student-test |
| Signed heading-cosine feature | About 95% of arcs have a reverse twin at the same distance |
| No class re-weighting | The probabilities become HMM emissions, and re-weighting distorts them |
| GroupKFold by trip; same 400k-row budget for every model | No leakage between folds; a fair comparison |
| Final model selected on **top-1 after HMM**, tie-break on log loss | This is what the evaluator measures. The Optuna-tuned model is better classifier-only but worse after HMM, so the untuned HistGradientBoosting is deployed (`FINAL_MODEL` in `src/config.py`) |
| HMM beta swept on student-test only | Allowed by the guidelines; evaluator trips are never used |

Results: see `report/results_tables.md`.

## 6. Team, contributions and AI-use disclosure

See `CONTRIBUTIONS.md`. Fill in the disclosure section there before submission.

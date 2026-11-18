# Technical report: section map

The PDF must follow the deliverables document, sections 2–7.

- **Numbers:** copy them from `report/results_tables.md`, which `scripts/report_tables.py` generates from the CSVs, so every value is verifiable.
- **Analysis text:** write it yourselves. The integrity checks look for AI-generated report prose.
- **Partitions:** state the data partition under every figure (train / student-test).

| Report section | Questions to answer | Evidence in the repo | Writer |
|---|---|---|---|
| **2.2 Dataset & EDA** | Q1 preprocessing; Q2 patterns, features and why | `outputs/eda/*.csv`, `outputs/data_quality/trip_label_quality.csv`, `src/labels.py`, `src/features.py` | M2 (EDA), M3 (preprocessing + features) |
| 2.2 required visuals | Missing-value analysis · GPS trajectory plot · road-network visualisation · Folium map | `trip_stats.csv` (missing_values, non_increasing_timestamps); `eda_trajectory_00000047.png`; `eda_network.png`; screenshot of `eda/maps/eda_trip_00000047.html` | M2 |
| **3.1 Comparison table** | Model, family, configuration, CV F1, top-1, continuity, test F1, log loss, inference time | `results_tables.md` §3.1 | M4 |
| **3.2 Per model (×5)** | A1–A2 suitability, strengths and limits · B5 hyperparameters · B6 loss · C7–C8 test results · C9 over/underfitting with evidence | `results_tables.md` §3.2, `cv_folds.csv`, `train_vs_validation_f1.png`, `mlp_training_curve.png` (MLP only), `confusion_matrices.png` | M4 (with M1–M3 each drafting one or two models) |
| **3.3 Visualisations** | CV F1 · test top-1 + F1 · log loss · confusion matrices · MLP curve only · model-selection plot | `outputs/figures/*.png`; selection: `all_models_classifier_vs_hmm.png` | M4 |
| **4 Bayesian optimisation** | Q1 parameters · Q2 ranges · Q3 objective · Q4 trials · Q5 best values · Q6 tuned better? Compare CV F1, test top-1, test F1, log loss, inference time | `outputs/tuning/*`, `results_tables.md` §4 | M4 |
| **5 HMM & Viterbi** | Q1 states · Q2 emission · Q3 transition · Q4 missing candidates / gaps · Q5 disconnected segments · Q6 improvement with results; classifier-only vs +HMM; where it didn't help | `src/hmm.py`, `outputs/hmm/*`, `results_tables.md` §5, `hmm_vs_classifier_per_trip.png` | M1 |
| **6 Folium** | GPS, network, reference, predicted; colours, legend, extent, layer control; one screenshot in the PDF | `outputs/maps/student_test_*.html`, `src/visualize.py` | M2 + M1 |
| **7 API, UI, deployment** | Screenshots of the working UI; screenshots of API responses in the receiving laptop's terminal | `report/screenshots/`, `api/main.py`, `ui/app.py`, `results_tables.md` §7 | M1 (API), M2 (UI) |
| AI-use disclosure | As required by the event policy | `CONTRIBUTIONS.md` | All |

**Screenshot checklist** (save in `report/screenshots/`)

1. UI with a trip selected and the metric cards visible.
2. UI map with the layer control open.
3. API terminal on laptop 1 showing the `===== request: trip … =====` block with timings and accuracy.
4. `http://<laptop-1-ip>:8000/docs` (optional).
5. One Folium map from `outputs/maps/` and the EDA map.

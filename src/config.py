"""Central configuration. Every script imports from here so all 4 laptops run identical settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"            # put train/ and student_test/ here (not committed)
PROC_DIR = ROOT / "data" / "processed"     # generated feature tables (not committed)
OUT_DIR = ROOT / "outputs"                 # CSVs, plots, maps (committed)
MODEL_DIR = ROOT / "models"                # saved models (committed if small)

PARTITIONS = ["train", "student_test"]     # evaluator_test is never touched by training code

SEED = 42
EARTH_R = 6_371_000.0                      # metres

# Candidate generation
CAND_RADIUS_M = 50.0      # EDA: true arc is inside 50 m for ~100% of points on clean trips
MAX_CANDIDATES = 15       # keep the K closest arcs per GPS point
DENSIFY_STEP_M = 5.0      # arcs are sampled every 5 m for the spatial index

# Label quality
LABEL_MAX_DIST_M = 50.0   # a point whose aligned route arc is further than this gets no label
BAD_TRIP_FRAC = 0.20      # trips with more than 20% unlabelled points are excluded from training

# HMM
HMM_BETA_M = 2.0          # swept 0.5-60 on student-test: 2.0 best (see outputs/hmm)
HMM_GAP_S = 30.0          # time gap that breaks the chain (decode segments independently)
HMM_UNREACHABLE_LOGP = -50.0
HMM_EMISSION_FLOOR = 1e-6

# Deployment (API on laptop 1). Override with environment variables, see README "Deployment".
# Selected on student-test top-1 AFTER HMM (outputs/hmm): untuned HGB 0.9534 vs Optuna-tuned 0.9460-0.9493,
# although the tuned model is better classifier-only (outputs/tuning). Tuned model kept as a documented experiment.
FINAL_MODEL = "hist_gradient_boosting"        # models/<FINAL_MODEL>.joblib
FALLBACK_MODEL = "hist_gradient_boosting"
EVAL_ROOT = ROOT / "data"                       # evaluation trips live in EVAL_ROOT / EVAL_PARTITION
EVAL_PARTITION = "evaluation"                   # put the 10 reserved trips in data/evaluation/<trip_id>/

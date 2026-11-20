"""PathSnap FastAPI backend - runs on LAPTOP 1.

Endpoints
  GET  /health            model + data status
  GET  /trips             the reserved evaluation trips (IDs + basic info only, no ground truth)
  POST /predict           {"trip_id": "..."} -> full pipeline -> predicted route + server-side accuracy

The reference routes stay on this laptop. They are read only by the evaluation step, after prediction,
and are never returned in a response (only the accuracy numbers computed from them are).

Run (from the repo root):
  uvicorn api.main:app --host 0.0.0.0 --port 8000
Environment variables (optional):
  PATHSNAP_EVAL_DIR   folder that contains one sub-folder per evaluation trip   (default data/evaluation)
  PATHSNAP_MODEL      model file stem inside models/                             (default from src/config.py)
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import EVAL_ROOT, EVAL_PARTITION, FINAL_MODEL, FALLBACK_MODEL, MODEL_DIR, HMM_BETA_M, OUT_DIR
from src.io_utils import list_trips, load_trip
from src.pipeline import predict_trip, evaluate_trip, route_polylines, nearby_network

# ---------- logging: console (shown to evaluators) + file (kept as evidence) ----------
log_dir = OUT_DIR / "api_logs"; log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "api.log")])
log = logging.getLogger("pathsnap")

eval_dir = Path(os.environ.get("PATHSNAP_EVAL_DIR", EVAL_ROOT / EVAL_PARTITION))
STATE = {}


def resolve_model():
    wanted = os.environ.get("PATHSNAP_MODEL", FINAL_MODEL)
    for name in (wanted, FALLBACK_MODEL):
        if (MODEL_DIR / f"{name}.joblib").exists():
            return name
    raise FileNotFoundError(f"no model file found in {MODEL_DIR} (looked for {wanted}, {FALLBACK_MODEL})")


@asynccontextmanager
async def lifespan(app):
    name = resolve_model()
    t0 = time.perf_counter()
    STATE["model"] = joblib.load(MODEL_DIR / f"{name}.joblib")
    STATE["model_name"] = name
    STATE["trips"] = list_trips(eval_dir.name, root=eval_dir.parent) if eval_dir.exists() else []
    log.info(f"model '{name}' loaded in {time.perf_counter() - t0:.2f}s | HMM beta={HMM_BETA_M} m")
    log.info(f"evaluation folder {eval_dir} -> {len(STATE['trips'])} trips: {STATE['trips']}")
    yield


app = FastAPI(title="PathSnap map-matching API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "?"
    log.info(f"{client} {request.method} {request.url.path} -> {response.status_code} "
             f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
    return response


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("model_name"), "hmm_beta_m": HMM_BETA_M,
            "evaluation_folder": str(eval_dir), "n_trips": len(STATE.get("trips", []))}


@app.get("/trips")
def trips():
    out = []
    for tid in STATE["trips"]:
        track = np.loadtxt(eval_dir / tid / f"{tid}.track", ndmin=2)
        out.append({"trip_id": tid, "n_points": int(len(track)),
                    "duration_min": round(float(track[-1, 2] - track[0, 2]) / 60, 1)})
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in STATE["trips"]:          # also blocks path tricks such as "../train/00000001"
        raise HTTPException(status_code=404, detail=f"unknown trip_id '{tid}'")
    log.info(f"===== request: trip {tid} =====")
    t_all = time.perf_counter()

    # 1) prediction: GPS + road network only (the .route file is not read here)
    t0 = time.perf_counter()
    trip = load_trip(eval_dir.name, tid, root=eval_dir.parent, with_route=False)
    load_s = time.perf_counter() - t0
    log.info(f"[{tid}] loaded {len(trip.track)} GPS points, {len(trip.nodes)} nodes, {len(trip.arcs)} arcs ({load_s:.2f}s)")
    res = predict_trip(trip, STATE["model"])
    tm = res["timings"]
    log.info(f"[{tid}] features {tm['candidates_features_s']:.2f}s ({res['n_candidate_rows']} candidate rows) | "
             f"classifier {tm['classifier_s']:.2f}s | HMM/Viterbi {tm['hmm_viterbi_s']:.2f}s")

    # 2) server-side evaluation against the reference route (after prediction is final)
    t0 = time.perf_counter()
    gt_trip = load_trip(eval_dir.name, tid, root=eval_dir.parent, with_route=True)
    metrics = evaluate_trip(gt_trip, res) if gt_trip.route is not None else None
    eval_s = time.perf_counter() - t0
    if metrics:
        log.info(f"[{tid}] top-1 accuracy: classifier {metrics['classifier_top1_accuracy']:.4f} -> "
                 f"HMM {metrics['hmm_top1_accuracy']:.4f} | continuity {metrics['hmm_route_continuity']:.4f} | "
                 f"route F1 {metrics['hmm_route']['f1']:.4f}")
    total_s = time.perf_counter() - t_all
    log.info(f"[{tid}] done in {total_s:.2f}s")

    step = max(1, len(trip.track) // 3000)          # thin GPS points for the map payload only
    return {
        "trip_id": tid, "model": STATE["model_name"], "hmm_beta_m": HMM_BETA_M,
        "n_points": int(len(trip.track)),
        "n_points_matched": int((res["hmm_pred"] >= 0).sum()),
        "metrics": metrics,
        "timings_s": {"load": load_s, **tm, "evaluation": eval_s, "total": total_s},
        "predicted_arcs": res["hmm_pred"].tolist(),                       # one arc id per GPS point (-1 = none)
        "geometry": {
            "gps": [[float(la), float(lo)] for lo, la in trip.track[::step, :2]],
            "hmm_route": route_polylines(trip, res["hmm_pred"]),
            "classifier_route": route_polylines(trip, res["clf_pred"]),
            "road_network": nearby_network(trip),
        },
    }

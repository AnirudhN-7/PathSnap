"""Calls the running API for EVERY trip it serves and records the results.
Proves the system works for all ten trips, not just one demo.
Output: outputs/api_logs/api_smoke_test.csv
Usage:  python -m scripts.api_smoke_test [--url http://<laptop-1-ip>:8000]"""
import argparse, time
import pandas as pd
import requests

from src.config import OUT_DIR


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    print("health:", requests.get(f"{url}/health", timeout=10).json())
    rows = []
    for t in requests.get(f"{url}/trips", timeout=30).json()["trips"]:
        t0 = time.perf_counter()
        r = requests.post(f"{url}/predict", json={"trip_id": t["trip_id"]}, timeout=600)
        rtt = time.perf_counter() - t0
        d = r.json(); m = d.get("metrics") or {}
        rows.append({"trip_id": t["trip_id"], "status": r.status_code, "n_points": d.get("n_points"),
                     "classifier_top1": m.get("classifier_top1_accuracy"), "hmm_top1": m.get("hmm_top1_accuracy"),
                     "hmm_continuity": m.get("hmm_route_continuity"), "hmm_route_f1": (m.get("hmm_route") or {}).get("f1"),
                     "api_compute_s": d.get("timings_s", {}).get("total"), "round_trip_s": rtt})
        print(f"trip {t['trip_id']}: HTTP {r.status_code}  top-1 {rows[-1]['classifier_top1']:.4f} -> "
              f"{rows[-1]['hmm_top1']:.4f}  ({rtt:.1f}s)")
    df = pd.DataFrame(rows)
    out = OUT_DIR / "api_logs"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "api_smoke_test.csv", index=False)
    ok = (df.status == 200).all()
    print(f"\n{len(df)} trips, all HTTP 200: {ok} | mean HMM top-1 {df.hmm_top1.mean():.4f} | "
          f"point-weighted {(df.hmm_top1 * df.n_points).sum() / df.n_points.sum():.4f}")


if __name__ == "__main__":
    main()

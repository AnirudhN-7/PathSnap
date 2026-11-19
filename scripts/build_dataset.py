"""Step 2: build candidate/feature tables for train and student_test.
Output: data/processed/{partition}_candidates.parquet, {partition}_points.parquet,
        outputs/data_quality/trip_label_quality.csv
Usage:  python -m scripts.build_dataset"""
import time
import pandas as pd
from joblib import Parallel, delayed

from src.config import PARTITIONS, PROC_DIR, OUT_DIR, BAD_TRIP_FRAC
from src.io_utils import list_trips, load_trip
from src.features import build_trip_table


def one(partition, tid):
    t0 = time.time()
    trip = load_trip(partition, tid)
    cand, pts = build_trip_table(trip)
    print(f"  {partition}/{tid}: {len(pts)} pts, {len(cand)} cand rows, {time.time()-t0:.1f}s", flush=True)
    return cand, pts


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data_quality").mkdir(parents=True, exist_ok=True)
    quality = []
    for part in PARTITIONS:
        res = Parallel(n_jobs=-1)(delayed(one)(part, tid) for tid in list_trips(part))
        cand = pd.concat([r[0] for r in res], ignore_index=True)
        pts = pd.concat([r[1] for r in res], ignore_index=True)
        q = pts.groupby("trip").agg(
            n_pts=("obs", "size"),
            frac_unlabelled=("label_valid", lambda s: 1 - s.mean()),
            median_label_dist_m=("label_dist", "median"),
            candidate_miss_rate=("label_in_cands", lambda s: 1 - s.mean()),
        ).reset_index()
        q["partition"] = part
        q["excluded_from_training"] = q["frac_unlabelled"] > BAD_TRIP_FRAC
        quality.append(q)
        cand.to_parquet(PROC_DIR / f"{part}_candidates.parquet", index=False)
        pts.to_parquet(PROC_DIR / f"{part}_points.parquet", index=False)
        print(f"{part}: {pts.trip.nunique()} trips, {len(pts)} points, {len(cand)} candidate rows, "
              f"positive rate {cand.y.mean():.3f}")
    q = pd.concat(quality)
    q.to_csv(OUT_DIR / "data_quality" / "trip_label_quality.csv", index=False)
    print("Excluded trips:", q.loc[q.excluded_from_training, "trip"].tolist())


if __name__ == "__main__":
    main()

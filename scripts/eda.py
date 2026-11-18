"""Step 1: dataset exploration (train + student_test). Evaluator-test data is never read.
Outputs (outputs/eda/): inventory.csv, trip_stats.csv, partition_summary.csv, data_quality_issues.csv,
        plots/*.png, maps/eda_trip_<id>.html
Usage:  python -m scripts.eda [--map-trip 00000001]"""
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.config import RAW_DIR, OUT_DIR, PARTITIONS
from src.io_utils import list_trips, load_trip
from src.geometry import ArcIndex
from src.plotting import plt, save, SERIES
from src.visualize import trip_map


def trip_stats(trip):
    xy, t = trip.xy, trip.track[:, 2]
    dt = np.diff(t); step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    spd = step / np.where(dt > 0, dt, np.nan)
    a = trip.arcs; n = len(trip.nodes)
    ncomp, _ = connected_components(coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n, n)), directed=False)
    deg = np.bincount(a.ravel(), minlength=n)
    key = set(map(tuple, a.tolist()))
    two_way = np.mean([(y, x) in key for x, y in a.tolist()])
    idx = ArcIndex(trip.node_xy, a)
    alen = idx.length
    d_route = np.nan
    if trip.route is not None:
        # nearest-route-arc distance per point (simple diagnostic; the training labels use the DP alignment)
        from src.geometry import project_to_segments
        RA, RB = idx.A[trip.route], idx.B[trip.route]
        d_route = np.array([project_to_segments(np.broadcast_to(p, (len(RA), 2)), RA, RB)[0].min() for p in xy])
    out = dict(
        partition=trip.partition, trip=trip.trip_id, n_points=len(t), duration_s=t[-1] - t[0],
        missing_values=int(np.isnan(trip.track).sum()), non_increasing_timestamps=int((dt <= 0).sum()),
        dt_median_s=np.median(dt), dt_max_s=dt.max(), frac_dt_1s=np.mean(dt == 1), gaps_over_10s=int((dt > 10).sum()),
        speed_median_ms=np.nanmedian(spd), speed_p99_ms=np.nanpercentile(spd, 99), jumps_over_50ms=int((spd > 50).sum()),
        lon_min=trip.track[:, 0].min(), lon_max=trip.track[:, 0].max(),
        lat_min=trip.track[:, 1].min(), lat_max=trip.track[:, 1].max(),
        extent_km=float(np.ptp(xy, axis=0).max() / 1000),
        n_nodes=n, n_arcs=len(a), arc_len_median_m=np.median(alen), two_way_arc_frac=two_way,
        connected_components=ncomp, isolated_nodes=int((deg == 0).sum()),
        route_len=len(trip.route) if trip.route is not None else 0,
        route_unique_arcs=len(set(trip.route.tolist())) if trip.route is not None else 0,
        route_contiguity=float(np.mean(a[trip.route[:-1], 1] == a[trip.route[1:], 0])) if trip.route is not None else np.nan,
        gps_to_route_median_m=float(np.nanmedian(d_route)), gps_to_route_p95_m=float(np.nanpercentile(d_route, 95)),
    )
    return out, dt, spd, d_route


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--map-trip", default="00000047")
    args = ap.parse_args()
    out = OUT_DIR / "eda"; plots = out / "plots"; maps = out / "maps"
    for d in (out, plots, maps): d.mkdir(parents=True, exist_ok=True)

    inv, rows, all_dt, all_spd, all_dr = [], [], [], [], []
    for part in PARTITIONS:
        for tid in list_trips(part):
            for ext in ["track", "nodes", "arcs", "route"]:
                f = RAW_DIR / part / tid / f"{tid}.{ext}"
                inv.append(dict(partition=part, trip=tid, file=ext, exists=f.exists(),
                                size_kb=f.stat().st_size / 1024 if f.exists() else 0,
                                rows=sum(1 for _ in open(f)) if f.exists() else 0))
            s, dt, spd, dr = trip_stats(load_trip(part, tid))
            rows.append(s)
            if part == "train":
                all_dt.append(dt); all_spd.append(spd); all_dr.append(dr)
            print(f"{part}/{tid}", flush=True)
    inv = pd.DataFrame(inv); st = pd.DataFrame(rows)
    inv.to_csv(out / "inventory.csv", index=False); st.to_csv(out / "trip_stats.csv", index=False)
    summ = st.drop(columns="trip").groupby("partition").agg(["count", "median", "min", "max"]).T
    summ.to_csv(out / "partition_summary.csv")

    issues = []
    for _, r in st.iterrows():
        if r.gps_to_route_p95_m > 50: issues.append((r.trip, r.partition, "reference route far from GPS", f"p95 {r.gps_to_route_p95_m:.0f} m"))
        if r.jumps_over_50ms > 5: issues.append((r.trip, r.partition, "implausible GPS jumps (>50 m/s)", int(r.jumps_over_50ms)))
        if r.dt_max_s > 60: issues.append((r.trip, r.partition, "large time gap", f"max {r.dt_max_s:.0f} s"))
        if r.connected_components > 100: issues.append((r.trip, r.partition, "fragmented road network", int(r.connected_components)))
        if r.route_contiguity < 1: issues.append((r.trip, r.partition, "route has non-contiguous steps", f"{r.route_contiguity:.3f}"))
    pd.DataFrame(issues, columns=["trip", "partition", "issue", "value"]).to_csv(out / "data_quality_issues.csv", index=False)

    dt = np.concatenate(all_dt); spd = np.concatenate(all_spd); dr = np.concatenate(all_dr)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].hist(st[st.partition == "train"].n_points, bins=30, color=SERIES[0], edgecolor="#fcfcfb")
    ax[0, 0].set(title="GPS points per trip (train)", xlabel="points", ylabel="trips")
    ax[0, 1].hist(np.clip(dt, 0, 30), bins=30, color=SERIES[0], edgecolor="#fcfcfb", log=True)
    ax[0, 1].set(title="Sampling interval (train, clipped at 30 s)", xlabel="seconds", ylabel="steps (log)")
    ax[1, 0].hist(np.clip(spd[np.isfinite(spd)], 0, 60), bins=60, color=SERIES[0], edgecolor="#fcfcfb")
    ax[1, 0].set(title="Step speed (train, clipped at 60 m/s)", xlabel="m/s", ylabel="steps")
    ax[1, 1].hist(np.clip(dr, 0, 60), bins=60, color=SERIES[0], edgecolor="#fcfcfb")
    ax[1, 1].set(title="GPS distance to nearest reference-route arc (train)", xlabel="metres (clipped at 60)", ylabel="points")
    fig.tight_layout(); save(fig, plots / "eda_distributions.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, part in enumerate(PARTITIONS):
        s = st[st.partition == part]
        ax[0].scatter(s.n_nodes, s.n_arcs, s=22, color=SERIES[i], label=part, edgecolor="#fcfcfb", linewidth=1)
        ax[1].scatter(s.n_arcs, s.connected_components, s=22, color=SERIES[i], label=part, edgecolor="#fcfcfb", linewidth=1)
    ax[0].set(title="Road-network size per trip", xlabel="nodes", ylabel="arcs", xscale="log", yscale="log")
    ax[1].set(title="Network fragmentation", xlabel="arcs", ylabel="connected components", xscale="log", yscale="log")
    ax[0].legend(); fig.tight_layout(); save(fig, plots / "eda_network.png")

    trip = load_trip("train", args.map_trip)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(trip.track[:, 0], trip.track[:, 1], ".", ms=2, color=SERIES[0])
    ax.set(title=f"GPS trajectory, trip {trip.trip_id} (train)", xlabel="longitude", ylabel="latitude")
    save(fig, plots / f"eda_trajectory_{trip.trip_id}.png")
    trip_map(trip).save(str(maps / f"eda_trip_{trip.trip_id}.html"))
    print(summ.to_string())


if __name__ == "__main__":
    main()

"""Reading raw trips and converting lon/lat to metres."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import RAW_DIR, EARTH_R


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray        # (n, 3) lon, lat, t
    nodes: np.ndarray        # (N, 2) lon, lat
    arcs: np.ndarray         # (M, 2) from_node, to_node  (row index = arc_id)
    route: np.ndarray | None # (R,) arc ids in driving order, None if not supplied
    lat0: float              # reference latitude for the local projection

    @property
    def xy(self):            # GPS points in metres
        return to_xy(self.track[:, 0], self.track[:, 1], self.lat0)

    @property
    def node_xy(self):
        return to_xy(self.nodes[:, 0], self.nodes[:, 1], self.lat0)


def to_xy(lon, lat, lat0):
    """Local equirectangular projection (metres). Each trip spans < 80 km, so the
    error vs. true geodesic distance is well under 1% - fine for 5-50 m GPS noise."""
    x = np.radians(lon) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat) * EARTH_R
    return np.column_stack([x, y])


def list_trips(partition, root=RAW_DIR):
    d = Path(root) / partition
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def load_trip(partition, trip_id, root=RAW_DIR, with_route=True) -> Trip:
    """root/partition/trip_id/trip_id.{track,nodes,arcs,route}. with_route=False never reads .route
    (used by the API's prediction step so ground truth cannot leak into predictions)."""
    d = Path(root) / partition / trip_id
    f = lambda ext: d / f"{trip_id}.{ext}"
    track = np.loadtxt(f("track"), dtype=float, ndmin=2)
    nodes = np.loadtxt(f("nodes"), dtype=float, ndmin=2)
    arcs = np.loadtxt(f("arcs"), dtype=np.int64, ndmin=2)
    route = np.loadtxt(f("route"), dtype=np.int64, ndmin=1) if with_route and f("route").exists() else None
    # timestamps were verified strictly increasing in EDA; sort defensively anyway
    track = track[np.argsort(track[:, 2], kind="stable")]
    return Trip(trip_id, partition, track, nodes, arcs, route, float(nodes[:, 1].mean()))

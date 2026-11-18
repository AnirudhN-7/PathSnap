"""Folium maps: GPS observations, nearby road network, reference route, predicted routes."""
import folium
import numpy as np
from scipy.spatial import cKDTree

LAYER_STYLE = {
    "network": dict(color="#9a9893", weight=1, opacity=0.6),
    "reference": dict(color="#008300", weight=6, opacity=0.65),
    "classifier": dict(color="#eb6834", weight=3, opacity=0.9),
    "hmm": dict(color="#6250d6", weight=3, opacity=0.95),
}


def _arc_lines(trip, arc_ids):
    nd = trip.nodes
    return [[(nd[a, 1], nd[a, 0]), (nd[b, 1], nd[b, 0])] for a, b in trip.arcs[arc_ids]]


def _route_runs(pred):
    """Collapse a per-point arc sequence into consecutive unique arcs."""
    pred = pred[pred >= 0]
    return pred[np.r_[True, pred[1:] != pred[:-1]]] if len(pred) else pred


def trip_map(trip, network_radius_m=150, clf_arcs=None, hmm_arcs=None, max_gps=4000):
    lat, lon = trip.track[:, 1], trip.track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # road network: only arcs with an endpoint near the trajectory (full networks are huge)
    near = cKDTree(trip.xy).query(trip.node_xy, distance_upper_bound=network_radius_m)[0] < np.inf
    arc_ids = np.where(near[trip.arcs[:, 0]] | near[trip.arcs[:, 1]])[0]
    g = folium.FeatureGroup(name=f"Road network (arcs within {network_radius_m} m)", show=True)
    for line in _arc_lines(trip, arc_ids):
        folium.PolyLine(line, **LAYER_STYLE["network"]).add_to(g)
    g.add_to(m)

    if trip.route is not None:
        g = folium.FeatureGroup(name="Reference route (.route)")
        for line in _arc_lines(trip, trip.route):
            folium.PolyLine(line, **LAYER_STYLE["reference"]).add_to(g)
        g.add_to(m)

    for key, arcs, label in [("classifier", clf_arcs, "Classifier-only prediction"),
                             ("hmm", hmm_arcs, "Classifier + HMM/Viterbi prediction")]:
        if arcs is not None:
            g = folium.FeatureGroup(name=label)
            for line in _arc_lines(trip, _route_runs(np.asarray(arcs))):
                folium.PolyLine(line, **LAYER_STYLE[key]).add_to(g)
            g.add_to(m)

    g = folium.FeatureGroup(name="GPS observations (.track)")
    step = max(1, len(lat) // max_gps)
    for i in range(0, len(lat), step):
        folium.CircleMarker((lat[i], lon[i]), radius=2, color="#2a78d6", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"obs {i}, t={trip.track[i, 2]:.0f}s").add_to(g)
    g.add_to(m)
    folium.Marker((lat[0], lon[0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((lat[-1], lon[-1]), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(_legend(trip, clf_arcs is not None, hmm_arcs is not None)))
    return m


def _legend(trip, clf, hmm):
    items = [("#2a78d6", "GPS observation", "dot"), ("#9a9893", "Road network", "line"),
             ("#008300", "Reference route", "line")]
    if clf: items.append(("#eb6834", "Classifier-only", "line"))
    if hmm: items.append(("#6250d6", "Classifier + HMM", "line"))
    rows = "".join(
        f'<div><span style="display:inline-block;width:18px;height:{8 if s=="dot" else 4}px;'
        f'border-radius:{4 if s=="dot" else 2}px;background:{c};margin-right:6px;vertical-align:middle"></span>{t}</div>'
        for c, t, s in items)
    return (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fcfcfb;padding:8px 12px;'
            f'border:1px solid #e4e3df;border-radius:6px;font:12px sans-serif;color:#0b0b0b">'
            f'<b>Trip {trip.trip_id} ({trip.partition})</b>{rows}</div>')

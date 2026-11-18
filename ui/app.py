"""PathSnap evaluator UI - runs on LAPTOP 2.

It only talks to the API over HTTP: it never reads the dataset, the model or the reference routes.
Flow: list the 10 evaluation trips (GET /trips) -> evaluator picks one -> POST /predict ->
show the accuracy computed by the API and draw GPS + predicted route on a Folium map.

Run:  streamlit run ui/app.py
API address: sidebar field, or set PATHSNAP_API_URL=http://<laptop-1-ip>:8000 before starting.
"""
import os
import time

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="PathSnap - Route Reconstruction", page_icon="🛰️", layout="wide")

COLORS = {"gps": "#2a78d6", "network": "#9a9893", "classifier": "#eb6834", "hmm": "#6250d6"}


# ---------------- sidebar: API connection ----------------
st.sidebar.header("API connection")
api_url = st.sidebar.text_input("API URL (laptop 1)", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")).rstrip("/")
timeout = st.sidebar.number_input("Request timeout (s)", 10, 600, 180)
try:
    h = requests.get(f"{api_url}/health", timeout=5).json()
    st.sidebar.success(f"Connected · model **{h['model']}** · HMM β = {h['hmm_beta_m']} m · {h['n_trips']} trips")
    connected = True
except Exception as e:  # noqa: BLE001 - show any connection problem to the user
    st.sidebar.error(f"Cannot reach the API at {api_url}\n\n{e}")
    connected = False
show_clf = st.sidebar.checkbox("Also draw classifier-only route", value=True)
show_net = st.sidebar.checkbox("Draw nearby road network", value=True)

st.title("PathSnap · GPS map-matching & route reconstruction")
st.caption("Pipeline on the API laptop: candidate road segments → feature extraction → "
           "HistGradientBoosting classifier → HMM/Viterbi decoding. Accuracy is computed on the API laptop; "
           "the reference route never leaves it.")

if not connected:
    st.stop()


@st.cache_data(ttl=60)
def get_trips(url):
    return requests.get(f"{url}/trips", timeout=30).json()["trips"]


trips = get_trips(api_url)
if not trips:
    st.warning("The API reports no evaluation trips. Check PATHSNAP_EVAL_DIR on laptop 1.")
    st.stop()

# ---------------- trip selection: the ten evaluation trips ----------------
st.subheader("1 · Select an evaluation trip")
labels = {t["trip_id"]: f"Trip {t['trip_id']}  ·  {t['n_points']:,} GPS points  ·  {t['duration_min']} min" for t in trips}
choice = st.radio("Evaluation trips", list(labels), format_func=labels.get, horizontal=False, label_visibility="collapsed")

if st.button("Run map-matching on the selected trip", type="primary"):
    with st.spinner(f"Sending trip {choice} to the API…"):
        t0 = time.perf_counter()
        r = requests.post(f"{api_url}/predict", json={"trip_id": choice}, timeout=timeout)
        rtt = time.perf_counter() - t0
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    st.session_state["result"] = r.json()
    st.session_state["rtt"] = rtt

res = st.session_state.get("result")
if not res:
    st.stop()

# ---------------- results ----------------
st.subheader(f"2 · Results for trip {res['trip_id']}")
m = res["metrics"]
if m:
    c = st.columns(5)
    c[0].metric("Top-1 accuracy (HMM)", f"{m['hmm_top1_accuracy']:.2%}",
                f"{(m['hmm_top1_accuracy'] - m['classifier_top1_accuracy']) * 100:+.1f} pts vs classifier")
    c[1].metric("Top-1 accuracy (classifier only)", f"{m['classifier_top1_accuracy']:.2%}")
    c[2].metric("Route continuity (HMM)", f"{m['hmm_route_continuity']:.2%}",
                f"{(m['hmm_route_continuity'] - m['classifier_route_continuity']) * 100:+.1f} pts")
    c[3].metric("Route F1 (HMM)", f"{m['hmm_route']['f1']:.3f}")
    c[4].metric("End-to-end time", f"{st.session_state['rtt']:.1f} s", f"API compute {res['timings_s']['total']:.1f} s",
                delta_color="off")
else:
    st.info("No reference route on the API laptop for this trip, so only the predicted route is shown.")

st.caption(f"Model **{res['model']}** · HMM β = {res['hmm_beta_m']} m · "
           f"{res['n_points_matched']:,} of {res['n_points']:,} GPS points matched")

# ---------------- map ----------------
st.subheader("3 · Predicted route")
g = res["geometry"]
lats = [p[0] for p in g["gps"]]; lons = [p[1] for p in g["gps"]]
fmap = folium.Map(location=[sum(lats) / len(lats), sum(lons) / len(lons)], tiles="OpenStreetMap", control_scale=True)
fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

if show_net and g["road_network"]:
    fg = folium.FeatureGroup(name="Road network near the trip")
    for seg in g["road_network"]:
        folium.PolyLine(seg, color=COLORS["network"], weight=1, opacity=0.6).add_to(fg)
    fg.add_to(fmap)
if show_clf:
    fg = folium.FeatureGroup(name="Classifier-only prediction", show=False)
    for seg in g["classifier_route"]:
        folium.PolyLine(seg, color=COLORS["classifier"], weight=3, opacity=0.9).add_to(fg)
    fg.add_to(fmap)
fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
for seg in g["hmm_route"]:
    folium.PolyLine(seg, color=COLORS["hmm"], weight=5, opacity=0.85).add_to(fg)
fg.add_to(fmap)
fg = folium.FeatureGroup(name="GPS observations")
for i, (la, lo) in enumerate(g["gps"]):
    folium.CircleMarker((la, lo), radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8, weight=0).add_to(fg)
fg.add_to(fmap)
folium.Marker(g["gps"][0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
folium.Marker(g["gps"][-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
folium.LayerControl(collapsed=False).add_to(fmap)

items = [(COLORS["gps"], "GPS observation", 8), (COLORS["hmm"], "Predicted route (HMM)", 4)]
if show_clf: items.append((COLORS["classifier"], "Classifier-only (toggle in layers)", 4))
if show_net: items.append((COLORS["network"], "Road network", 4))
legend = "".join(f'<div><span style="display:inline-block;width:18px;height:{hgt}px;border-radius:3px;background:{col};'
                 f'margin-right:6px;vertical-align:middle"></span>{txt}</div>' for col, txt, hgt in items)
fmap.get_root().html.add_child(folium.Element(
    f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;padding:8px 12px;'
    f'border:1px solid #ddd;border-radius:6px;font:12px sans-serif"><b>Trip {res["trip_id"]}</b>{legend}</div>'))
map_html = fmap.get_root().render()
if hasattr(st, "iframe"):                      # Streamlit >= 1.52
    st.iframe(map_html, height=620)
else:                                          # older Streamlit
    components.html(map_html, height=620)

# ---------------- details ----------------
with st.expander("Timing breakdown (seconds, measured on the API laptop)"):
    st.table({k: [round(v, 3)] for k, v in res["timings_s"].items()})
with st.expander("Raw API response (metrics only)"):
    st.json({k: v for k, v in res.items() if k not in ("geometry", "predicted_arcs")})

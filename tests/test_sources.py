"""
Offline tests: API parsers against response shapes taken from each provider's docs,
plus the network helpers. No keys or internet needed.

Run:  python -m pytest -q tests      (or: python tests/test_sources.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sources import railkit, railradar, openmeteo, osm


RAILKIT_HISTORY = {   # shape from railkit.in/docs/train-history
    "trainNo": "12301", "trainName": "RAJDHANI EXPRES", "journeyDate": "11-06-2026",
    "stations": [
        {"stationCode": "HWH", "stationName": "HOWRAH JN", "platform": "9",
         "arrival": {"scheduled": "SRC", "actual": "SRC"},
         "departure": {"scheduled": "16:50 11-Jun", "actual": "16:50 11-Jun", "delay": "On Time"}},
        {"stationCode": "ASN", "stationName": "ASANSOL JN.", "distanceKm": "200",
         "arrival": {"scheduled": "18:47 11-Jun", "actual": "19:03 11-Jun", "delay": "16 Min"},
         "departure": {"scheduled": "18:49 11-Jun", "actual": "19:05 11-Jun*", "delay": "16 Min"}},
        {"stationCode": "NDLS", "stationName": "NEW DELHI", "distanceKm": "1449",
         "arrival": {"scheduled": "10:05 12-Jun", "actual": "10:13 12-Jun", "delay": "8 Min"},
         "departure": {"scheduled": "DSTN", "actual": "DSTN"}},
    ]}

RAILRADAR_LIVE = {    # shape from railradar.in/docs/live-train-status
    "trainNumber": "12301", "trainName": "Rajdhani", "status": "running", "delayMinutes": 12,
    "currentLocation": {"stationCode": "ASN", "sequence": 2, "status": "departed",
                        "segmentProgress": 0.5, "speedKmh": 110, "isActualPosition": True},
    "nextHalt": {"stationCode": "NDLS"},
    "route": [{"stationCode": "HWH", "lat": 22.58, "lng": 88.34},
              {"stationCode": "ASN", "lat": 23.68, "lng": 86.98},
              {"stationCode": "NDLS", "lat": 28.64, "lng": 77.22}]}

RAILRADAR_MAP = [     # shape from railradar.in/docs/legacy-live-map
    {"train_number": "12468", "train_name": "Leelan", "type": "SF", "current_lat": 26.90,
     "current_lng": 75.42, "current_station": "JOB", "next_station": "HDA", "curr_distance": 37,
     "next_distance": 46, "mins_since_dep": 40},
    {"train_number": "12301", "train_name": "Rajdhani", "type": "RAJ", "current_lat": 23.68,
     "current_lng": 86.98, "current_station": "ASN", "next_station": "DHN", "curr_distance": 200,
     "next_distance": 259, "mins_since_dep": 135}]


def test_railkit_history_rows():
    df = railkit.history_to_rows(RAILKIT_HISTORY)
    assert list(df.station_code) == ["HWH", "ASN"]
    assert df.current_delay_min.tolist() == [0, 16]
    assert df.distance_remaining_km.tolist() == [1449, 1249]
    # exit arrival 10:13 12-Jun, departed ASN 19:05 -> 908 min actual, 916 scheduled
    assert df.actual_remaining_time_min.iloc[1] == 908
    assert df.remaining_delay_min.iloc[1] == 908 - 916


def test_railkit_corridor_clip():
    df = railkit.history_to_rows(RAILKIT_HISTORY, corridor=["ASN", "NDLS"])
    assert df is None            # only one departing station left -> not enough to build rows


def test_railradar_live_state():
    s = railradar.live_state(RAILRADAR_LIVE)
    assert s["delay_min"] == 12 and s["station_code"] == "ASN" and s["next_station"] == "NDLS"
    assert abs(s["lat"] - (23.68 + 0.5 * (28.64 - 23.68))) < 1e-9


def test_railradar_snapshot_filter():
    pts = railradar.snapshot_points(RAILRADAR_MAP)
    assert pts.train_no.tolist() == ["12468"]     # Howrah-area train is outside the bounding box


def test_openmeteo_merge():
    rows = pd.DataFrame({"station_code": ["JP"], "actual_departure_ts": [pd.Timestamp("2026-08-01 16:29")]})
    w = pd.DataFrame({"time": pd.to_datetime(["2026-08-01 16:00", "2026-08-01 17:00"]), "station_code": "JP",
                      **{c: [1.0, 2.0] for c in openmeteo.WEATHER_COLS}})
    out = openmeteo.merge_weather(rows, w)
    assert out.temperature_2m.iloc[0] == 1.0


def test_osm_match():
    o = pd.DataFrame({"osm_id": [1, 2], "name": ["Jaipur Junction", "Phulera"], "ref": ["", "FL"],
                      "lat": [26.92, 26.87], "lon": [75.79, 75.25]})
    st = pd.DataFrame({"station_code": ["JP", "FL"], "station_name": ["JAIPUR Jn.", "PHULERA"]})
    m = osm.match_codes(st, o)
    assert m["FL"][2] == "osm:ref" and m["JP"][2] == "osm:name"


def test_dbscan_cluster():
    from models.dbscan_network import cluster_points
    out = cluster_points([26.90, 26.91, 27.50], [75.40, 75.41, 74.00])
    size = out[1] if isinstance(out, tuple) else out
    assert list(size) == [2, 2, 1]


def test_fusion_weights_sum_to_one():
    from models.fusion import inverse_mae_weights
    w = inverse_mae_weights({"xgb": 5.0, "kf": 7.0, "gcn": 6.0})
    assert abs(sum(w.values()) - 1) < 1e-12 and w["xgb"] > w["gcn"] > w["kf"]


def test_cascade_zero_delay_changes_nothing():
    from models.cascade import propagate
    plan = pd.DataFrame({"train_no": [1, 2], "journey_date": pd.to_datetime(["2026-08-30"] * 2),
                         "train_name": ["A", "B"], "station_sequence": [1, 1], "station_code": ["JP", "JP"],
                         "next_station": ["FL", "FL"], "t": [600.0, 612.0]})      # B follows A by 12 min
    assert propagate(plan, 1, "JP", 0).empty
    # A +10 -> leaves 610; B must keep 6 min headway -> 616, i.e. pushed 4 min
    assert propagate(plan, 1, "JP", 10).set_index("train_no").added_delay_min.to_dict() == {1: 10.0, 2: 4.0}


def test_osm_path_follows_track():
    from pipeline.build_osm_paths import path_on_ways, route_path
    # a curved track from A (26.90, 75.00) to B (26.90, 75.20) bulging north, plus a spur
    curve = [(26.90, 75.00), (26.95, 75.05), (26.97, 75.10), (26.95, 75.15), (26.90, 75.20)]
    ways = [{"id": 1, "nodes": [], "geometry": curve[:3]}, {"id": 2, "nodes": [], "geometry": curve[2:]},
            {"id": 3, "nodes": [], "geometry": [(26.97, 75.10), (27.05, 75.10)]}]
    line = path_on_ways(ways, (26.90, 75.00), (26.90, 75.20), (26.8, 74.9, 27.1, 75.3))
    assert line is not None and max(p[0] for p in line) > 26.96          # went over the curve
    assert all(p[0] < 27.0 for p in line)                                # did not take the spur
    # nothing within 2 km -> None (caller falls back to a straight line)
    assert path_on_ways(ways, (25.0, 75.0), (25.0, 75.2), (24.9, 74.9, 25.1, 75.3)) is None
    full = route_path(["A", "B"], {"A>B": [[1, 1], [2, 2]]}, {})
    back = route_path(["B", "A"], {"A>B": [[1, 1], [2, 2]]}, {})
    assert full == [[1, 1], [2, 2]] and back == [[2, 2], [1, 1]]


def test_openmeteo_batch_and_kind(monkeypatch=None):
    import sources.openmeteo as om
    calls = []

    def fake(url, **kw):
        calls.append(kw["params"])
        n = len(kw["params"]["latitude"].split(","))
        return [{"hourly": {"time": ["2026-08-01T00:00", "2026-08-01T01:00"], "visibility": [300, 15000]}}] * n
    orig = om.get_json
    om.get_json = fake
    try:
        st = pd.DataFrame({"station_code": ["JP", "FL", "AII"], "lat": [26.9, 26.87, 26.45], "lon": [75.8, 75.25, 74.64]})
        out = om.recent_hourly(st, ["visibility"], past_days=40, batch=2)
    finally:
        om.get_json = orig
    assert len(calls) == 2 and len(out) == 6 and set(out.station_code) == {"JP", "FL", "AII"}
    assert om.weather_kind(0, 0, 0.3) == "fog" and om.weather_kind(0, 0, 0.1) == "dense_fog"
    assert om.weather_kind(63, 1.0, 10) == "rain" and om.weather_kind(95, 0, 10) == "heavy_rain"
    assert om.weather_kind(2, 0, 10) == "cloudy" and om.weather_kind(0, 0, None) == "clear"


if __name__ == "__main__":  # noqa
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

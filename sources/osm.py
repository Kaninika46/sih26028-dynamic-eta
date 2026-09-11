"""
OpenStreetMap via the Overpass API (no key needed, please keep requests small).

  stations(bbox)       railway=station|halt nodes with name / railway:ref / ref tags
  match_codes(...)     WTT station codes -> OSM coordinates (by ref tag, then by name)
  track_geojson(bbox)  railway=rail lines as GeoJSON for the map
"""
import json
import re
from difflib import SequenceMatcher

import pandas as pd

from config import CORRIDOR_BBOX
from sources.http import get_json

OVERPASS = "https://overpass-api.de/api/interpreter"


def _query(q, ttl_s=None):
    return get_json(OVERPASS, provider="osm", method="POST", data={"data": q}, ttl_s=ttl_s, timeout=120)


def stations(bbox=CORRIDOR_BBOX):
    s, w, n, e = bbox
    q = f"""[out:json][timeout:90];
    ( node["railway"~"^(station|halt)$"]({s},{w},{n},{e});
      way["railway"="station"]({s},{w},{n},{e}); );
    out center tags;"""
    js = _query(q)
    rows = []
    for el in js.get("elements", []):
        t = el.get("tags", {})
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lon = el.get("lon", (el.get("center") or {}).get("lon"))
        rows.append(dict(osm_id=el["id"], name=t.get("name:en") or t.get("name"),
                         ref=(t.get("railway:ref") or t.get("ref") or "").upper(), lat=lat, lon=lon))
    return pd.DataFrame(rows)


def _norm(s):
    s = re.sub(r"\b(jn|junction|halt|road|rd|city)\b\.?", "", str(s).lower())
    return re.sub(r"[^a-z]", "", s)


def match_codes(sched_stations, osm_df):
    """sched_stations: DataFrame(station_code, station_name). Returns code -> (lat, lon, how)."""
    out = {}
    for _, r in sched_stations.iterrows():
        hit = osm_df[osm_df.ref == r.station_code]
        if len(hit):
            out[r.station_code] = (hit.lat.iloc[0], hit.lon.iloc[0], "osm:ref")
            continue
        if osm_df.empty:
            continue
        sc = osm_df.name.fillna("").map(lambda n: SequenceMatcher(None, _norm(n), _norm(r.station_name)).ratio())
        if sc.max() >= 0.85:
            k = sc.idxmax()
            out[r.station_code] = (osm_df.lat[k], osm_df.lon[k], "osm:name")
    return out


def track_geojson(bbox=CORRIDOR_BBOX):
    s, w, n, e = bbox
    q = f"""[out:json][timeout:120];
    way["railway"="rail"]["usage"~"main|branch"]({s},{w},{n},{e});
    out geom;"""
    js = _query(q)
    feats = [{"type": "Feature", "properties": {"osm_id": el["id"]},
              "geometry": {"type": "LineString", "coordinates": [[p["lon"], p["lat"]] for p in el["geometry"]]}}
             for el in js.get("elements", []) if el.get("geometry")]
    return {"type": "FeatureCollection", "features": feats}


# ----------------------------------------------------------------------------
# Railway geometry between consecutive stations (for drawing real track paths)
# ----------------------------------------------------------------------------
def rail_ways_in_boxes(boxes):
    """boxes: list of (s, w, n, e). One Overpass request for all of them (union).
    Returns list of ways: {"id", "nodes", "geometry": [(lat, lon), ...]}."""
    parts = "".join(f'way["railway"="rail"]({s:.4f},{w:.4f},{n:.4f},{e:.4f});' for s, w, n, e in boxes)
    js = _query(f"[out:json][timeout:180];({parts});out geom;")
    ways = []
    for el in (js or {}).get("elements", []):
        if el.get("type") == "way" and el.get("geometry"):
            ways.append({"id": el["id"], "nodes": el.get("nodes") or [],
                         "geometry": [(p["lat"], p["lon"]) for p in el["geometry"]]})
    return ways

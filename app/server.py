"""
Backend for the RUNTIME web UI (app/ui/index.html).

The UI calls, on the same host:
  GET /api/coords  -> {"STATION NAME": [lat, lon], ...}
  GET /api/trains  -> [{id, name, from, to, stations, stopIdx, predMin, confLo, confHi,
                        weather, tempC, visibilityKm, breakdown, override}, ...]
Extra endpoints (for demos / future UI buttons):
  GET /api/health
  GET /api/cascade?train=12468&station=JP&delay=30   -> trains pushed back by that delay
  GET /api/trains?t=18:30                             -> replay a chosen time of day

What is shown: the 40 trains replayed on a held-out test day (the models never trained on
it) at the current time of day. predMin / confLo / confHi = predicted delay at the train's
final station = current delay + fused remaining-delay prediction (XGBoost v2 +
1D Kalman + GCN-LSTM) and its 80% range. breakdown = XGBoost contribution groups (for the
XGBoost v2, all 40 trains) scaled to the predicted delay.

Run:  python app/server.py      then open http://localhost:8000
      REPLAY_DATE=2026-08-30 python app/server.py
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import DATA, OUT
from sources import supa
from models.common import KEY, load_data, split

UI = Path(__file__).resolve().parent / "ui" / "index.html"


class Engine:
    def __init__(self):
        from models.xgb_branch import load as load_features
        df = load_features()          # dataset + network + engineered features (as the models see it)
        _, _, test = split(df)
        days = sorted(test.journey_date.dt.date.unique())
        want = os.getenv("REPLAY_DATE")
        self.day = pd.Timestamp(want).date() if want else days[len(days) // 2]
        self.df = df
        f = pd.read_csv(OUT / "fused_predictions.csv")
        f["journey_date"] = pd.to_datetime(f.journey_date, format="%d-%m-%Y")
        self.fused = f.set_index(KEY).sort_index()
        feat = OUT / "dataset_with_network_features.csv"
        self.cong = None
        if feat.exists():
            c = pd.read_csv(feat, usecols=KEY + ["congestion_ahead_30"])
            c["journey_date"] = pd.to_datetime(c.journey_date, format="%d-%m-%Y")
            self.cong = c.set_index(KEY).congestion_ahead_30
        self.coords = (df.groupby("station_name")[["latitude", "longitude"]].first().dropna()
                       .round(5).apply(list, axis=1).to_dict())
        # Open-Meteo weather (pipeline.add_weather) and OSM track paths (pipeline.build_osm_paths)
        self.wx = None
        wx = DATA / "weather_hourly.csv"
        if wx.exists():
            w = pd.read_csv(wx, parse_dates=["time"])
            self.wx = w.set_index(["station_code", "time"]).sort_index()
        pj = DATA / "osm_paths.json"
        self.paths = json.loads(pj.read_text()) if pj.exists() else {}
        self.code_xy = (df.groupby("station_code")[["latitude", "longitude"]].first().dropna()
                        .round(5).apply(list, axis=1).to_dict())
        from models import xgb_branch
        self.xgb = xgb_branch
        self.xgb_model = xgb_branch.load_model()
        self.known = set(xgb_branch.known_trains())

    def breakdown(self, row, f, cong, wkind=None):
        out = []
        if int(row.train_no) in self.known:
            r = self.xgb.reasons(self.xgb_model, pd.DataFrame([row])).iloc[0]
            for k, v in r.items():
                label = f"Weather ({wkind.replace('_', ' ')})" if k == "Weather" and wkind else k
                out.append([label, round(abs(float(v)), 1)])
        else:
            kf = f.get("kf_pred", np.nan)
            out.append(["Running delay carried", round(abs(float(row.current_delay_min)) * 0.1, 1)])
            if kf == kf:
                out.append(["Historical pattern", round(abs(float(kf)), 1)])
        if cong and cong > 0.5:
            out.append(["Section congestion", round(float(cong), 1)])
        # keep the weather row whenever the model uses weather, even when its effect is small,
        # so the app can always say what the weather contributed
        wx = [b for b in out if b[0].startswith("Weather")]
        rest = sorted([b for b in out if not b[0].startswith("Weather") and b[1] >= 0.5], key=lambda b: -b[1])
        return (rest[:3] + wx) if wx else rest[:4]

    def weather(self, code, t):
        if self.wx is None:
            return None, None, None
        try:
            r = self.wx.loc[(code, t.floor("h"))]
        except KeyError:
            return None, None, None
        from sources.openmeteo import weather_kind
        vis = r.get("visibility_km")
        vis = None if vis is None or vis != vis else round(float(vis), 1)
        temp = r.get("temperature_2m")
        return (weather_kind(r.get("weather_code"), r.get("rain"), vis),
                None if temp != temp else round(float(temp), 1), vis)

    def track(self, codes, k):
        from pipeline.build_osm_paths import route_path
        full = route_path(codes, self.paths, self.code_xy)
        if len(full) > 600:
            full = [full[int(i)] for i in np.linspace(0, len(full) - 1, 600)]
        seg = route_path(codes[k:k + 2], self.paths, self.code_xy) if k + 1 < len(codes) else []
        return full, seg

    @staticmethod
    def scale(parts, total):
        """Show each cause as its share of the predicted delay (parts sum to predMin)."""
        s = sum(v for _, v in parts)
        if s <= 0 or total <= 0:
            return []
        return [[k, round(total * v / s, 1)] for k, v in parts]

    def trains(self, hhmm=None):
        now = pd.Timestamp.now()
        if hhmm:
            h, m = map(int, hhmm.split(":"))
            now = now.replace(hour=h, minute=m)
        t = pd.Timestamp(self.day) + pd.Timedelta(hours=now.hour, minutes=now.minute)
        win = self.df[(self.df.journey_date >= pd.Timestamp(self.day) - pd.Timedelta(days=3))
                      & (self.df.journey_date <= pd.Timestamp(self.day))]
        out = []
        for (tn, jd), g in win.groupby(["train_no", "journey_date"]):
            g = g.sort_values("station_sequence").reset_index(drop=True)
            last = g.iloc[-1]
            end = last.dep_ts + pd.Timedelta(minutes=float(last.actual_remaining_time_min))
            if not (g.dep_ts.iloc[0] <= t < end):
                continue
            k = int((g.dep_ts <= t).sum()) - 1
            row = g.loc[k]
            key = (tn, jd, row.station_sequence)
            if key not in self.fused.index:
                continue
            f = self.fused.loc[key]
            f = f.iloc[0] if isinstance(f, pd.DataFrame) else f
            cur = float(row.current_delay_min)
            cong = float(self.cong.get(key, 0)) if self.cong is not None else 0.0
            names = g.station_name.tolist()
            codes = g.station_code.tolist()
            # real scheduled clock time at every stop, so the timeline shows the timetable
            # plus the predicted delay instead of "now + a minute per stop"
            sched_times = [ts.strftime("%H:%M") if pd.notna(ts) else None for ts in g.sched_dep_ts]
            wkind, temp, vis = self.weather(row.station_code, t)
            path, seg = self.track(codes, k)
            if k + 1 < len(g):
                run = max((g.sched_dep_ts[k + 1] - g.sched_dep_ts[k]).total_seconds() / 60, 1.0)
                prog = float(np.clip((t - row.dep_ts).total_seconds() / 60 / run, 0.05, 0.95))
            else:
                prog = 0.5
            out.append({
                "id": str(tn), "name": str(row.train_name).title(),
                "from": names[0], "to": names[-1], "stations": names, "stopIdx": k,
                "schedTimes": sched_times,
                "predMin": round(max(0.0, cur + float(f.fused_pred)), 1),
                "confLo": round(max(0.0, cur + float(f.fused_low)), 1),
                "confHi": round(max(0.0, cur + float(f.fused_high)), 1),
                "currentDelayMin": round(cur, 1),
                "etaFinal": (row.dep_ts + pd.Timedelta(minutes=float(row.scheduled_remaining_time_min
                                                                     + f.fused_pred))).strftime("%H:%M"),
                "weather": wkind, "tempC": temp, "visibilityKm": vis,
                "path": path, "segPath": seg, "progress": round(prog, 3),
                "breakdown": self.scale(self.breakdown(row, f, cong, wkind),
                                        max(0.0, cur + float(f.fused_pred))),
                "override": None,
            })
        out.sort(key=lambda x: -x["predMin"])
        return out

    def live_train_list(self):
        return sorted(self.df.train_no.astype(int).unique())

    def cascade(self, train, station, delay):
        from models.cascade import plan_for_day, propagate
        plan = plan_for_day(self.df, self.day)
        return propagate(plan, int(train), station, float(delay)).to_dict(orient="records")


LIVE = os.getenv("LIVE", "").lower() in ("1", "true", "yes")
ENGINE = None

# ------------------------------------------------------------------ access control
# Which screens (tabs) and actions each role may use. Enforced here on the server for every
# privileged request, and a second time by row-level security inside Supabase.
PERMS = {
    "guest":      {"views": ["passenger"], "actions": []},
    "passenger":  {"views": ["passenger"], "actions": []},
    "station":    {"views": ["station", "passenger"], "actions": ["log"]},
    "controller": {"views": ["control", "station", "passenger", "analytics"],
                   "actions": ["override", "cascade", "audit", "log"]},
    "admin":      {"views": ["control", "station", "passenger", "analytics"],
                   "actions": ["override", "cascade", "audit", "log"]},
}
PORTAL_ROLES = {"ctrl": {"controller", "admin"}, "stn": {"station", "controller", "admin"},
                "pax": {"passenger", "station", "controller", "admin"}}
SEVERITY_BUMP = [("Critical", 45), ("High", 30), ("Moderate", 18), ("Low", 8)]
_ov_cache = {"t": 0, "rows": []}


def active_overrides():
    """Active overrides from the last 6 h (server reads with the secret key; cached 15 s)."""
    if not supa.enabled() or not supa.SECRET:
        return []
    if time.time() - _ov_cache["t"] > 15:
        since = (pd.Timestamp.utcnow() - pd.Timedelta(hours=6)).isoformat()
        try:
            _ov_cache["rows"] = supa.select("overrides", {"active": "eq.true", "created_at": f"gte.{since}",
                                                          "order": "created_at.desc"})
        except supa.AuthError:
            _ov_cache["rows"] = []
        _ov_cache["t"] = time.time()
    return _ov_cache["rows"]


def apply_overrides(trains, role):
    by_train = {}
    for o in active_overrides():
        by_train.setdefault(str(o["train_no"]), []).append(o)
    for t in trains:
        for o in by_train.get(t["id"], [])[:1]:                      # newest override wins
            b = float(o["bump_min"])
            t["predMin"] = round(t["predMin"] + b, 1)
            t["confLo"] = round(t["confLo"] + b * 0.7, 1)
            t["confHi"] = round(t["confHi"] + b * 1.3, 1)
            t["breakdown"] = [[f"{o['incident_type']} (override)", b]] + t["breakdown"]
            t["status"] = "amber" if "Low" in o["severity"] else "red"
            t["override"] = {"type": o["incident_type"], "loc": o.get("location") or "",
                             "sev": o["severity"], "note": o.get("note") or "",
                             "time": pd.Timestamp(o["created_at"]).tz_convert("Asia/Kolkata").strftime("%H:%M"),
                             "by": "Control room" if role in ("guest", "passenger") else "Controller"}
    return trains


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        return self._send(json.dumps(obj, default=str), code=code)

    def _user(self):
        """Signed-in user from the Authorization header, or a guest."""
        h = self.headers.get("Authorization", "")
        token = h[7:] if h.startswith("Bearer ") else ""
        if not supa.enabled():                                   # demo mode: no database configured
            return {"role": "controller", "demo": True}, ""
        if not token:
            return {"role": "guest"}, ""
        return supa.verify(token), token

    def _require(self, action):
        user, token = self._user()
        if action not in PERMS[user["role"]]["actions"]:
            raise supa.AuthError(f"your role ({user['role']}) is not allowed to {action}", 403)
        return user, token

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(UI.read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/config":                           # public: what the browser needs
                return self._json({"auth": supa.enabled(), "supabaseUrl": supa.URL,
                                   "supabaseKey": supa.PUBLISHABLE, "perms": PERMS})
            if u.path == "/api/health":
                return self._json({"ok": True, "replay_day": str(ENGINE.day), "auth": supa.enabled(),
                                   "live": LIVE})
            if u.path == "/api/coords":
                return self._json(ENGINE.coords)
            if u.path == "/api/me":
                user, _ = self._user()
                if user["role"] == "guest":
                    raise supa.AuthError("not signed in")
                return self._json({**user, **PERMS[user["role"]]})
            if u.path == "/api/trains":
                user, _ = self._user()
                live = LIVE or q.get("live") == "1"
                if live:
                    try:
                        from models.live_eta import live_cards
                        cards = live_cards(ENGINE.live_train_list())
                        if cards:
                            return self._json(apply_overrides(cards, user["role"]))
                        print("live feed returned no running trains - falling back to replay")
                    except Exception as e:
                        print(f"live mode failed ({e}) - falling back to replay")
                return self._json(apply_overrides(ENGINE.trains(q.get("t")), user["role"]))
            if u.path == "/api/cascade":
                self._require("cascade")
                return self._json(ENGINE.cascade(q["train"], q["station"], q.get("delay", 30)))
            if u.path == "/api/audit":
                user, token = self._require("audit")
                if user.get("demo"):
                    return self._json([])
                return self._json(supa.select("audit_log", {"select": "created_at,role,action,train_no,detail",
                                                            "order": "created_at.desc", "limit": "50"}, token=token))
            return self._json({"error": "not found"}, 404)
        except supa.AuthError as e:
            return self._json({"error": str(e)}, e.status)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/override":
                user, token = self._require("override")
                b = self._body()
                sev = str(b.get("severity", "Moderate"))
                bump = next((v for k, v in SEVERITY_BUMP if k in sev), 18)
                row = {"train_no": int(b["train_no"]), "incident_type": str(b.get("type", "Incident"))[:80],
                       "location": str(b.get("location", ""))[:80], "severity": sev[:40],
                       "note": str(b.get("note", ""))[:500], "bump_min": bump}
                if not user.get("demo"):
                    saved = supa.insert("overrides", [row], token=token)       # RLS checks the role again
                    supa.insert("audit_log", [{"role": user["role"], "action": "override",
                                               "train_no": row["train_no"], "detail": row}], token=token)
                    _ov_cache["t"] = 0
                    return self._json({"ok": True, "override": saved[0]})
                return self._json({"ok": True, "override": row, "demo": True})
            if u.path == "/api/log":                                          # staff sign-ins, board opened
                user, token = self._require("log")
                b = self._body()
                if not user.get("demo"):
                    supa.insert("audit_log", [{"role": user["role"], "action": str(b.get("action", "event"))[:60],
                                               "detail": b.get("detail")}], token=token)
                return self._json({"ok": True})
            return self._json({"error": "not found"}, 404)
        except supa.AuthError as e:
            return self._json({"error": str(e)}, e.status)
        except (KeyError, ValueError) as e:
            return self._json({"error": f"bad request: {e}"}, 400)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ENGINE = Engine()
    port = int(os.getenv("PORT", "8000"))
    if not supa.enabled():
        print("WARNING: SUPABASE_URL / SUPABASE_PUBLISHABLE_KEY not set -> DEMO MODE, no login, "
              "every screen is open. Set them in .env to turn on authentication.")
    print(("LIVE mode: RailRadar feed through the trained models."
           if LIVE else f"Replaying test day {ENGINE.day}.") + f" Open http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()

"""
End-to-end test of login + roles: the real app/server.py talks to tests/fake_supabase.py.
Checks that passengers / station terminals cannot use control-room endpoints, that
controllers can, and that an override shows up for everyone.

Run:  python tests/test_auth.py     (takes ~30 s: the server loads the models)
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.update(SUPABASE_URL="http://127.0.0.1:9999", SUPABASE_PUBLISHABLE_KEY="sb_publishable_test",
                  SUPABASE_SECRET_KEY="sb_secret_test")

import fake_supabase  # noqa: E402

fake_supabase.start(9999)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("server", ROOT / "app" / "server.py")
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)
srv.ENGINE = srv.Engine()
http = srv.ThreadingHTTPServer(("127.0.0.1", 8765), srv.Handler)
threading.Thread(target=http.serve_forever, daemon=True).start()
BASE = "http://127.0.0.1:8765"


def call(path, token=None, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                                 method="POST" if body is not None else "GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, cond):
    print(("ok  " if cond else "FAIL"), name)
    assert cond, name


cfg = call("/api/config")[1]
check("config says auth on and never leaks the secret key",
      cfg["auth"] and "sb_secret" not in json.dumps(cfg))
check("guest can read trains", call("/api/trains?t=10:30")[0] == 200)
check("guest has no profile", call("/api/me")[0] == 401)
check("bad token rejected", call("/api/me", "tok-forged")[0] == 401)

me = {k: call("/api/me", f"tok-{k}")[1] for k in ("pax", "stn", "ctrl", "admin")}
check("passenger sees only passenger view", me["pax"]["views"] == ["passenger"])
check("station sees station + passenger", me["stn"]["views"] == ["station", "passenger"])
check("controller sees control room", "control" in me["ctrl"]["views"])

ov = {"train_no": 14662, "type": "Signal failure", "location": "JP", "severity": "High", "note": "test"}
check("guest cannot override", call("/api/override", None, ov)[0] == 403)
check("passenger cannot override", call("/api/override", "tok-pax", ov)[0] == 403)
check("station cannot override", call("/api/override", "tok-stn", ov)[0] == 403)
check("passenger cannot read audit log", call("/api/audit", "tok-pax")[0] == 403)
check("station cannot read audit log", call("/api/audit", "tok-stn")[0] == 403)
check("passenger cannot run cascade", call("/api/cascade?train=14662&station=JP&delay=30", "tok-pax")[0] == 403)
check("station can log an event", call("/api/log", "tok-stn", {"action": "station_board_opened"})[0] == 200)

code, res = call("/api/override", "tok-ctrl", ov)
check("controller can override (saved in database)", code == 200 and res["override"]["bump_min"] == 30)
check("controller reads audit log", any(a["action"] == "override" for a in call("/api/audit", "tok-ctrl")[1]))
check("controller runs cascade", call("/api/cascade?train=14662&station=JP&delay=30", "tok-ctrl")[0] == 200)

time.sleep(0.2)
trains = call("/api/trains?t=10:30", "tok-pax")[1]
t = next((x for x in trains if x["id"] == "14662"), None)
if t:
    check("override visible to passengers, staff identity hidden",
          t["override"] and t["override"]["by"] == "Control room" and t["breakdown"][0][1] == 30)
else:
    print("skip 14662 not running at 10:30")
print("\nall auth checks passed")

"""
A tiny fake of the Supabase endpoints the server uses, with the same row-level-security
rules as supabase/schema.sql. Lets tests/test_auth.py check the whole login / role flow
without internet.  Tokens: tok-pax, tok-stn, tok-ctrl, tok-admin.
"""
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

SECRET = "sb_secret_test"
USERS = {"tok-pax": ("u1", "pax@test.in", "passenger"), "tok-stn": ("u2", "stn@test.in", "station"),
         "tok-ctrl": ("u3", "ctrl@test.in", "controller"), "tok-admin": ("u4", "admin@test.in", "admin")}
DB = {"overrides": [], "audit_log": [], "stations": [], "eta_predictions": [], "train_movements": [], "live_positions": [], "admin_users": [], "profile_updates": []}


def who(h):
    if h.get("apikey") == SECRET:
        return "service"
    tok = (h.get("Authorization") or "")[7:]
    return USERS.get(tok)


class H(BaseHTTPRequestHandler):
    def send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        w = who(self.headers)
        if u.path == "/auth/v1/user":
            return self.send({"id": w[0], "email": w[1]}) if isinstance(w, tuple) else self.send({"msg": "bad jwt"}, 401)
        table = u.path.rsplit("/", 1)[-1]
        if table == "profiles":
            if not isinstance(w, tuple) or q.get("id") != f"eq.{w[0]}":
                return self.send([])                                   # RLS: only own row
            return self.send([{"role": w[2], "station_code": None, "full_name": w[1], "employee_id": None}])
        if table in DB:
            if w == "service" or (isinstance(w, tuple) and w[2] in ("controller", "admin")):
                return self.send(list(reversed(DB[table])))
            return self.send([])                                       # RLS hides rows
        self.send({"msg": "no table"}, 404)

    def do_PATCH(self):
        rows = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if who(self.headers) != "service":
            return self.send({"message": "permission denied"}, 403)
        DB["profile_updates"].append(rows)
        self.send([rows])

    def do_POST(self):
        u = urlparse(self.path)
        table = u.path.rsplit("/", 1)[-1]
        rows = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        w = who(self.headers)
        if w == "service":
            if u.path == "/auth/v1/admin/users":
                DB["admin_users"].append(rows)
                return self.send({"id": f"new-{len(DB['admin_users'])}", "email": rows["email"]})
            DB[table] += rows if isinstance(rows, list) else [rows]
            return self.send(rows, 201)
        allowed = {"overrides": ("controller", "admin"), "audit_log": ("station", "controller", "admin")}
        if not (isinstance(w, tuple) and w[2] in allowed.get(table, ())):
            return self.send({"message": "new row violates row-level security policy"}, 403)
        now = datetime.now(timezone.utc).isoformat()
        out = [{**r, "id": len(DB[table]) + i + 1, "created_at": now, "active": True} for i, r in enumerate(rows)]
        DB[table] += out
        self.send(out, 201)

    def log_message(self, *a):
        pass


def start(port=9999):
    s = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s

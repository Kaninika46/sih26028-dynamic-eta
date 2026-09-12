"""
Create a user and give them a role (staff accounts cannot sign themselves up as staff).

  python -m pipeline.create_user controller@nwr.in "StrongPass#1" controller --employee NWR-8841
  python -m pipeline.create_user jp.board@nwr.in  "StrongPass#2" station --station JP
  python -m pipeline.create_user me@example.com   "StrongPass#3" admin

Needs SUPABASE_SECRET_KEY. Passengers usually sign up themselves in the app.
"""
import argparse

from sources import supa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("email")
    ap.add_argument("password")
    ap.add_argument("role", choices=["passenger", "station", "controller", "admin"])
    ap.add_argument("--station", default=None, help="station code for station terminals, e.g. JP")
    ap.add_argument("--employee", default=None, help="employee ID for staff")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    user = supa.admin_create_user(a.email, a.password, a.name)
    uid = user["id"]
    # the database trigger created the profile as 'passenger'; promote it with the secret key
    supa.update("profiles", {"id": f"eq.{uid}"},
                {"role": a.role, "station_code": a.station, "employee_id": a.employee})
    print(f"created {a.email} ({uid}) as {a.role}")


if __name__ == "__main__":
    main()

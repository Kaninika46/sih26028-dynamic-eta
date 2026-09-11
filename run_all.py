"""
Whole pipeline on the team's 40-train data (about 10-15 minutes on a laptop CPU).

    python run_all.py
    python app/server.py        -> open http://localhost:8000  (the team's UI)

XGBoost v2 (models/xgb_branch.py) is retrained on all 40 trains after the network features exist.
"""
import subprocess
import sys
import time

# needs internet; if it fails the pipeline continues (UI then hides weather / draws straight lines)
OPTIONAL = {"pipeline.add_weather", "pipeline.build_osm_paths"}
STEPS = ["pipeline.prepare_data", "pipeline.add_weather", "pipeline.build_osm_paths", "models.kalman_1d", "models.dbscan_network",
         "models.kalman_2d", "models.xgb_branch", "models.gcn_lstm", "models.fusion"]
for s in STEPS:
    t = time.time()
    print(f"\n=== {s} ===", flush=True)
    if subprocess.run([sys.executable, "-m", s]).returncode:
        if s in OPTIONAL:
            print(f"(optional step {s} failed - continuing without it)")
            continue
        sys.exit(f"step failed: {s}")
    print(f"({time.time() - t:.0f}s)", flush=True)
subprocess.run([sys.executable, "tests/test_sources.py"])
print("\nDone. Start the UI with:  python app/server.py   (then open http://localhost:8000)")

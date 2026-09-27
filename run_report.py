"""
run_report.py

One canonical record per training run, written next to the loss curve and GIF.

    save_run_report(model_folder, settings, metrics, run_name, master_csv)

    -> writes   <model_folder>/run_report.csv   (section,key,value rows)
    -> updates  master_csv                      (one wide row per run)

No dependencies beyond the standard library. The CSV opens directly in Excel.
"""

import csv
import os
from datetime import datetime

REPORT_NAME = "run_report.csv"


def save_run_report(model_folder, settings, metrics, run_name, master_csv):
    """
    settings   : dict of run configuration (anything you put in gets saved)
    metrics    : dict of results; list values (phase_center / phase_angle) are
                 split into one row per phase
    run_name   : the run's row name in the master CSV (a rerun replaces it)
    master_csv : path of the master CSV; its columns grow as new keys appear
    """
    os.makedirs(model_folder, exist_ok=True)

    rows = [("meta", "run_name", run_name),
            ("meta", "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]

    for k in sorted(settings.keys()):
        rows.append(("settings", str(k), str(settings[k])))

    for k, v in metrics.items():
        if isinstance(v, (list, tuple)):          # phase_center / phase_angle
            for name, vi in zip(("airborne", "contact", "settled"), v):
                rows.append(("metrics", f"{k}_{name}", repr(float(vi))))
        else:
            rows.append(("metrics", str(k), repr(float(v))))

    report_path = os.path.join(model_folder, REPORT_NAME)
    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        w.writerows(rows)
    print(f"[run_report] saved {report_path}")

    # ---- master file: one wide row per run, columns grow as needed ----
    flat = {"run_name": run_name, "timestamp": rows[1][2]}
    for section, key, value in rows[2:]:
        flat[f"{section}.{key}"] = value

    existing, fieldnames = [], []
    if os.path.exists(master_csv):
        with open(master_csv, newline="") as f:
            rdr = csv.DictReader(f)
            fieldnames = list(rdr.fieldnames or [])
            existing = [row for row in rdr]
    for k in flat:
        if k not in fieldnames:
            fieldnames.append(k)
    existing = [r for r in existing if r.get("run_name") != run_name]  # replace reruns
    existing.append(flat)
    with open(master_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(existing)
    print(f"[run_report] master updated: {master_csv} ({len(existing)} runs)")

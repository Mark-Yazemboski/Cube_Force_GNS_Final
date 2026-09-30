"""
This file provides functionality to save and update run reports for experiments.
This is essential for keeping track of experiment runs and their results.
"""

import csv
import os
from datetime import datetime

# Name of the CSV file used for run reports.
# These are individual run reports saved alongside the model.
# This is not the master CSV; it only contains the report for this specific run.
REPORT_NAME = "run_report.csv"


#This function saves the run report for a specific experiment run. 
# It writes a detailed CSV in the model folder and updates the master CSV.
#inputs:
# model_folder : directory where the model and its run report will be saved
# settings     : dictionary containing the run configuration
# metrics      : dictionary containing the results of the run
# run_name     : name of the run, used as the row identifier in the master CSV
# master_csv   : path to the master CSV file that aggregates all runs
def save_run_report(model_folder, settings, metrics, run_name, master_csv):

    # Ensure the model folder exists before saving the report.
    os.makedirs(model_folder, exist_ok=True)

    # Prepare the rows for the CSV report. Each row is a tuple of (section, key, value).
    rows = [("meta", "run_name", run_name),
            ("meta", "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]

    # Add settings to the report rows.
    for k in sorted(settings.keys()):
        rows.append(("settings", str(k), str(settings[k])))

    # Add metrics to the report rows. Lists/tuples are expanded into separate rows.
    for k, v in metrics.items():

        # Check if the metric value is a list or tuple, indicating multiple sub-metrics.
        if isinstance(v, (list, tuple)):         

            # Expand each element of the list/tuple into separate rows with descriptive names.
            for name, vi in zip(("airborne", "contact", "settled"), v):
                rows.append(("metrics", f"{k}_{name}", repr(float(vi))))
        else:
            rows.append(("metrics", str(k), repr(float(v))))

    # Save the individual run report CSV in the model folder.
    report_path = os.path.join(model_folder, REPORT_NAME)

    # Write the CSV file with the prepared rows.
    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        w.writerows(rows)
    print(f"[run_report] saved {report_path}")

    # ---- master file: one wide row per run, columns grow as needed ----

    # Flatten the rows into a single dictionary for the master CSV.
    flat = {"run_name": run_name, "timestamp": rows[1][2]}

    # Convert the remaining rows into key-value pairs for the master CSV.
    for section, key, value in rows[2:]:
        flat[f"{section}.{key}"] = value

    # Read the existing master CSV if it exists, and update the fieldnames to include any new keys.
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

    # Write the updated master CSV with all runs, including the current one.
    with open(master_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(existing)
    print(f"[run_report] master updated: {master_csv} ({len(existing)} runs)")

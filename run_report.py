"""
run_report.py

One canonical record per training run, written next to the loss curve and GIF.

WRITER (call at the end of a run script):
    from run_report import save_run_report
    metrics = evaluate_model(...)            # patched to return full aggregates
    slopes  = plot_phase_error_curves(...)   # patched to return the slope rows
    save_run_report(model_folder, settings, metrics, slopes)

    -> writes  <model_folder>/run_report.csv        (section,key,value rows)
    -> appends <models>/all_runs_master.csv         (one wide row per run)

READER (paste-block generator):
    python run_report.py Paper_W_16_decay
    python run_report.py models/Paper_W_16_decay models/Paper_W_8_none ...

    One folder  -> full formatted block (identical layout to the console
                   output you've been copy-pasting), ready to paste.
    Many folders -> compact comparison table first, then each full block.

No dependencies beyond the standard library. The CSV opens directly in Excel.
"""

import csv
import os
import sys
from datetime import datetime

REPORT_NAME = "run_report.csv"
MASTER_NAME = "all_runs_master.csv"


# ======================================================================
# WRITER
# ======================================================================
def save_run_report(model_folder, settings, metrics, slopes,
                    run_name=None, master_csv="auto"):
    """
    settings : dict of run configuration (anything you put in gets saved)
    metrics  : dict returned by evaluate_model (patched version)
    slopes   : list of dicts returned by plot_phase_error_curves (patched),
               each with keys metric, phase, median_frames, slope, max_mean_error
    master_csv : "auto" -> <parent of model_folder>/all_runs_master.csv
                 None   -> skip the master file
                 path   -> use that path
    """
    os.makedirs(model_folder, exist_ok=True)
    run_name = run_name or os.path.basename(os.path.normpath(model_folder))

    rows = [("meta", "run_name", run_name),
            ("meta", "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]

    for k in sorted(settings.keys()):
        rows.append(("settings", str(k), str(settings[k])))

    for k, v in (metrics or {}).items():
        if isinstance(v, (list, tuple)):          # phase_center / phase_angle
            for name, vi in zip(("airborne", "contact", "settled"), v):
                rows.append(("metrics", f"{k}_{name}", repr(float(vi))))
        else:
            rows.append(("metrics", str(k), repr(float(v))))

    for r in (slopes or []):
        base = f"{r['metric'].split()[0].lower()}_{r['phase']}"   # center_airborne
        rows.append(("slopes", f"{base}_median_frames", str(int(r["median_frames"]))))
        rows.append(("slopes", f"{base}_slope", repr(float(r["slope"]))))
        rows.append(("slopes", f"{base}_max_mean_error", repr(float(r["max_mean_error"]))))

    report_path = os.path.join(model_folder, REPORT_NAME)
    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        w.writerows(rows)
    print(f"[run_report] saved {report_path}")

    # ---- master file: one wide row per run, columns grow as needed ----
    if master_csv is not None:
        if master_csv == "auto":
            master_csv = os.path.join(os.path.dirname(os.path.normpath(model_folder)),
                                      MASTER_NAME)
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


# ======================================================================
# READER
# ======================================================================
def _load(model_folder):
    path = os.path.join(model_folder, REPORT_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No {REPORT_NAME} in {model_folder}\n"
            f"  contents: {sorted(os.listdir(model_folder))[:12]}")
    data = {"meta": {}, "settings": {}, "metrics": {}, "slopes": {}}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["section"], {})[row["key"]] = row["value"]
    return data


def _f(d, key, default=float("nan")):
    try:
        return float(d[key])
    except (KeyError, ValueError, TypeError):
        return default


def format_report(model_folder):
    d = _load(model_folder)
    m, s = d["metrics"], d["slopes"]
    out = []
    bar = "=" * 68
    out.append(bar)
    out.append(f"RUN REPORT: {d['meta'].get('run_name', model_folder)}"
               f"    (saved {d['meta'].get('timestamp', '?')})")
    out.append(bar)

    out.append("--- Settings ---")
    for k, v in sorted(d["settings"].items()):
        out.append(f"{k} = {v}")

    out.append("")
    out.append("--- Test Set Metrics ---")
    out.append(f"Center Error ( / width):   {_f(m,'center_error'):.4f} "
               f"\u00b1 {_f(m,'center_error_std'):.4f}")
    out.append(f"Angle Error (degrees):     {_f(m,'angle_error_deg'):.4f} "
               f"\u00b1 {_f(m,'angle_error_std'):.4f}")
    out.append(f"Floor Penetration (/ width):         {_f(m,'floor_penetration'):.4f} "
               f"\u00b1 {_f(m,'floor_penetration_std'):.4f}")
    out.append("")
    out.append("--- Phase breakdown (airborne / contact / settled) ---")
    out.append(f"Center (/width): {_f(m,'phase_center_airborne'):.4f} / "
               f"{_f(m,'phase_center_contact'):.4f} / {_f(m,'phase_center_settled'):.4f}")
    out.append(f"Angle (deg):     {_f(m,'phase_angle_airborne'):.4f} / "
               f"{_f(m,'phase_angle_contact'):.4f} / {_f(m,'phase_angle_settled'):.4f}")

    out.append("")
    for metric, label in (("center", "Center Error"), ("angle", "Angle Error ")):
        for phase in ("airborne", "contact", "settled"):
            b = f"{metric}_{phase}"
            if f"{b}_slope" in s:
                out.append(f"{label} | {phase:<9s} | "
                           f"Median Frames: {int(_f(s, b + '_median_frames', 0)):3d} | "
                           f"Slope: {_f(s, b + '_slope'):8.5f} | "
                           f"Max Mean Error: {_f(s, b + '_max_mean_error'):8.5f}")
    out.append(bar)
    return "\n".join(out)


def comparison_table(folders):
    cols = [("center", lambda m, s: f"{_f(m,'center_error'):.4f}\u00b1{_f(m,'center_error_std'):.3f}"),
            ("angle", lambda m, s: f"{_f(m,'angle_error_deg'):.2f}\u00b1{_f(m,'angle_error_std'):.2f}"),
            ("c-slope", lambda m, s: f"{_f(s,'center_contact_slope'):.5f}"),
            ("a-slope", lambda m, s: f"{_f(s,'angle_contact_slope'):.3f}")]
    key_settings = ["multistep", "impact_weight", "scheduler", "noise_scale"]
    lines = []
    header = f"{'run':<38}" + "".join(f"{c[0]:>15}" for c in cols) \
             + "".join(f"{k:>14}" for k in key_settings)
    lines.append(header)
    lines.append("-" * len(header))
    for folder in folders:
        try:
            d = _load(folder)
        except FileNotFoundError:
            lines.append(f"{os.path.basename(os.path.normpath(folder)):<38}  <no report>")
            continue
        m, s, st = d["metrics"], d["slopes"], d["settings"]
        row = f"{d['meta'].get('run_name',''):<38}"
        row += "".join(f"{fn(m, s):>15}" for _, fn in cols)
        row += "".join(f"{st.get(k, st.get(k.capitalize(), '-')):>14}"
                       for k in key_settings)
        lines.append(row)
    return "\n".join(lines)


def _resolve(arg):
    if os.path.isdir(arg):
        return arg
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "models", arg), os.path.join(here, arg),
                 os.path.join("models", arg)):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"Could not find a model folder for '{arg}'")


if __name__ == "__main__":
    test_names = ["Paper_Error_Plots" ]
    folders = [_resolve(a) for a in test_names]
    print(f"[run_report] found {len(folders)} folders: {folders}")
    if len(folders) > 1:
        print(comparison_table(folders))
        print()
    for f in folders:
        print(format_report(f))
        print()

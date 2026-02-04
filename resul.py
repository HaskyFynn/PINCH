#!/usr/bin/env python3
"""
Exp1 Robustness plots (minimal set)

Default outputs (2 figures):
  1) stable accuracy by condition (mean across trials, error bars across trials)
  2) total latency by condition (boxplot, per-frame latency)

Also exports:
  - plots/metrics_by_condition.csv
  - plots/share_summary.csv  (small, human-friendly)
Optional:
  - plots/share_sample.csv   (small sample of rows for one chosen condition)

Input:
  logs/session_YYYYMMDD_HHMMSS/*_per_frame.csv

Run:
  python plot_exp1_min.py --session-dir .\\logs\\session_20260114_120500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_png_pdf(fig, out_base: Path):
    ensure_dir(out_base.parent)
    fig.tight_layout()
    fig.savefig(str(out_base.with_suffix(".png")), dpi=200)
    fig.savefig(str(out_base.with_suffix(".pdf")))
    plt.close(fig)

def p95(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    if v.size == 0:
        return float("nan")
    return float(np.percentile(v, 95))

def require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

def cond_label(light: str, dist: str) -> str:
    return f"{light} | {dist}"

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


# -------------------------
# Load
# -------------------------

def load_session(session_dir: Path) -> pd.DataFrame:
    csvs = sorted(session_dir.glob("*_per_frame.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_per_frame.csv found in: {session_dir}")

    dfs = []
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        df["__file"] = p.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    required = [
        "trial_id", "frame_idx", "lighting", "distance",
        "slot_idx", "true_id",
        "detected", "accepted_id", "stable_id",
        "is_correct_stable", "is_wrong_accepted", "is_miss",
        "lat_total_ms", "proc_fps"
    ]
    require_cols(df, required)

    # Types
    for c in ["frame_idx","slot_idx","detected","is_correct_stable","is_wrong_accepted","is_miss","lat_total_ms","proc_fps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["trial_id","lighting","distance","true_id","accepted_id","stable_id"]:
        df[c] = df[c].astype(str)

    df["cond"] = df.apply(lambda r: cond_label(str(r["lighting"]), str(r["distance"])), axis=1)
    return df


# -------------------------
# Metrics
# -------------------------

def compute_metrics(df: pd.DataFrame):
    # Slot metrics per trial (each true_id is a slot identity)
    g_slot = df.groupby(["trial_id","cond","lighting","distance","true_id"], dropna=False)

    slot = g_slot.agg(
        frames=("frame_idx","count"),
        stable_acc=("is_correct_stable","mean"),
        miss_rate=("is_miss","mean"),
        wrong_rate=("is_wrong_accepted","mean"),
        recognized_stable=("stable_id", lambda s: float(np.mean((s != "unknown")))),
    ).reset_index()

    # Trial metrics (average across slots)
    g_trial = slot.groupby(["trial_id","cond","lighting","distance"], dropna=False)
    trial = g_trial.agg(
        n_slots=("true_id","count"),
        stable_acc=("stable_acc","mean"),
        miss_rate=("miss_rate","mean"),
        wrong_rate=("wrong_rate","mean"),
        recognized_stable=("recognized_stable","mean"),
    ).reset_index()

    # Latency is duplicated per slot row; dedupe per frame
    frame = df.drop_duplicates(subset=["trial_id","frame_idx"]).copy()
    g_lat = frame.groupby(["trial_id","cond","lighting","distance"], dropna=False)

    lat = g_lat.agg(
        frames=("frame_idx","count"),
        lat_total_ms_med=("lat_total_ms","median"),
        lat_total_ms_p95=("lat_total_ms", p95),
        fps_med=("proc_fps","median"),
    ).reset_index()

    trial_all = pd.merge(trial, lat, on=["trial_id","cond","lighting","distance"], how="left")

    # Condition summary across trials
    g_cond = trial_all.groupby(["cond","lighting","distance"], dropna=False)
    cond_mean = g_cond.mean(numeric_only=True).reset_index()
    cond_std  = g_cond.std(numeric_only=True).reset_index()

    return slot, trial_all, cond_mean, cond_std, frame


def make_share_summary(cond_mean: pd.DataFrame, cond_std: pd.DataFrame) -> pd.DataFrame:
    # Small human-friendly sheet
    keep = ["cond","stable_acc","miss_rate","wrong_rate","recognized_stable","lat_total_ms_med","lat_total_ms_p95","fps_med"]
    m = cond_mean.copy()
    s = cond_std.copy()

    for c in keep:
        if c not in m.columns:
            m[c] = np.nan
        if c not in s.columns:
            s[c] = np.nan

    out = pd.DataFrame()
    out["cond"] = m["cond"].astype(str)
    out["stable_acc_mean"] = m["stable_acc"]
    out["stable_acc_std"] = s["stable_acc"]
    out["miss_rate_mean"] = m["miss_rate"]
    out["wrong_rate_mean"] = m["wrong_rate"]
    out["recognized_stable_mean"] = m["recognized_stable"]
    out["lat_total_ms_med_mean"] = m["lat_total_ms_med"]
    out["lat_total_ms_p95_mean"] = m["lat_total_ms_p95"]
    out["fps_med_mean"] = m["fps_med"]

    # Sort by stable accuracy (descending)
    out = out.sort_values("stable_acc_mean", ascending=False).reset_index(drop=True)
    return out


# -------------------------
# Plots (minimal)
# -------------------------

def plot_stable_acc(cond_mean: pd.DataFrame, cond_std: pd.DataFrame, out_base: Path):
    dfm = cond_mean[["cond","stable_acc"]].copy()
    dfs = cond_std[["cond","stable_acc"]].copy()

    dfm = dfm.sort_values("stable_acc", ascending=False)
    dfs = dfs.set_index("cond").reindex(dfm["cond"]).reset_index()

    x = dfm["cond"].astype(str).to_list()
    y = dfm["stable_acc"].to_numpy()
    yerr = dfs["stable_acc"].to_numpy()

    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, y, yerr=yerr)
    ax.set_ylim(0, 1.0)
    ax.set_title("Exp1: Stable ID accuracy by condition")
    ax.set_ylabel("stable accuracy (stable_id == true_id)")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    save_png_pdf(fig, out_base)

def plot_latency_box(frame_df: pd.DataFrame, out_base: Path):
    df = frame_df.copy()
    df["lat_total_ms"] = pd.to_numeric(df["lat_total_ms"], errors="coerce")
    df = df[np.isfinite(df["lat_total_ms"])]

    if df.empty:
        return

    order = df.groupby("cond")["lat_total_ms"].median().sort_values().index.to_list()
    data = [df.loc[df["cond"] == c, "lat_total_ms"].to_numpy() for c in order]

    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(data, labels=order, showfliers=False)
    ax.set_title("Exp1: Total latency by condition (per processed frame)")
    ax.set_ylabel("lat_total_ms")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    save_png_pdf(fig, out_base)

def plot_miss_rate(cond_mean: pd.DataFrame, cond_std: pd.DataFrame, out_base: Path):
    dfm = cond_mean[["cond","miss_rate"]].copy()
    dfs = cond_std[["cond","miss_rate"]].copy()

    dfm = dfm.sort_values("miss_rate", ascending=True)
    dfs = dfs.set_index("cond").reindex(dfm["cond"]).reset_index()

    x = dfm["cond"].astype(str).to_list()
    y = dfm["miss_rate"].to_numpy()
    yerr = dfs["miss_rate"].to_numpy()

    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, y, yerr=yerr)
    ax.set_ylim(0, min(1.0, float(np.nanmax(y)) * 1.3 + 1e-6) if np.isfinite(np.nanmax(y)) else 1.0)
    ax.set_title("Exp1: Miss rate by condition")
    ax.set_ylabel("miss rate (no assigned track in slot)")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    save_png_pdf(fig, out_base)


# -------------------------
# Sampling for sharing values
# -------------------------

def export_sample(df: pd.DataFrame, out_csv: Path, cond: str = "", n_rows: int = 300):
    """
    Export a small sample with only the most relevant columns.
    Useful to paste values here or sanity-check distribution.

    cond example: "Bright | Near"
    """
    keep_cols = [
        "trial_id","cond","lighting","distance",
        "slot_idx","true_id","stable_id","accepted_id",
        "is_correct_stable","is_wrong_accepted","is_miss",
        "lat_total_ms","proc_fps"
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""

    d = df.copy()
    if cond:
        d = d[d["cond"].astype(str) == cond]

    if d.empty:
        return False

    # Random sample to keep it small
    d = d.sample(n=min(n_rows, len(d)), random_state=7)[keep_cols]
    d.to_csv(out_csv, index=False)
    return True


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", type=str, required=True)
    ap.add_argument("--plot-miss", action="store_true", help="Add 3rd plot: miss rate by condition.")
    ap.add_argument("--export-sample", action="store_true", help="Export plots/share_sample.csv for one condition.")
    ap.add_argument("--cond", type=str, default="", help='Condition string, example: "Bright | Near"')
    ap.add_argument("--sample-rows", type=int, default=300)
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(session_dir)

    plots_dir = session_dir / "plots"
    ensure_dir(plots_dir)

    df = load_session(session_dir)
    slot, trial_all, cond_mean, cond_std, frame_df = compute_metrics(df)

    # Save numeric tables
    trial_all.to_csv(plots_dir / "metrics_by_trial.csv", index=False)
    cond_tbl = cond_mean.merge(cond_std, on=["cond","lighting","distance"], suffixes=("_mean","_std"))
    cond_tbl.to_csv(plots_dir / "metrics_by_condition.csv", index=False)

    share = make_share_summary(cond_mean, cond_std)
    share.to_csv(plots_dir / "share_summary.csv", index=False)

    # Print a compact table to console (top lines)
    print("\nExp1 summary (share_summary.csv):")
    with pd.option_context("display.max_rows", 50, "display.max_columns", 50, "display.width", 140):
        print(share)

    # Minimal plots (2 by default)
    plot_stable_acc(cond_mean, cond_std, plots_dir / "fig1_stable_accuracy_by_condition")
    plot_latency_box(frame_df, plots_dir / "fig2_latency_box_by_condition")

    if args.plot_miss:
        plot_miss_rate(cond_mean, cond_std, plots_dir / "fig3_miss_rate_by_condition")

    if args.export_sample:
        cond = args.cond.strip()
        ok = export_sample(df, plots_dir / "share_sample.csv", cond=cond, n_rows=int(args.sample_rows))
        if not ok:
            print("\n[warn] Could not export sample. Condition not found or empty.")
        else:
            print("\nSample exported:", (plots_dir / "share_sample.csv").resolve())

    print("\nSaved plots to:", plots_dir.resolve())
    print("Key share file:", (plots_dir / "share_summary.csv").resolve())
    print("Figures (PNG + PDF):")
    print("  - fig1_stable_accuracy_by_condition")
    print("  - fig2_latency_box_by_condition")
    if args.plot_miss:
        print("  - fig3_miss_rate_by_condition")


if __name__ == "__main__":
    main()

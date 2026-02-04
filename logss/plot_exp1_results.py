#!/usr/bin/env python3
"""
PINCH Exp1 Plotter (Robustness)

Reads *_per_frame.csv logs and produces 3 paper-ready plots:
1) Stable ID accuracy heatmap (lighting x distance)
2) Latency p95 heatmap (lighting x distance)
3) Miss-rate heatmap (lighting x distance)

Outputs:
- condition_summary.csv
- figures as PDF (and PNG) into <session>/plots/

Usage examples:
  python plot_exp1_results.py --session ./logs/session_YYYYMMDD_HHMMSS
  python plot_exp1_results.py --logs-root ./logs

Notes:
- This script assumes the per-frame schema produced by exp1.py (your robustness runner).
- Latency is deduplicated by (trial_id, frame_idx) because latency values are repeated per slot.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REQ_COLS = {
    "trial_id","frame_idx","lighting","distance","slot_idx","true_id",
    "is_correct_stable","is_miss","is_wrong_accepted",
    "lat_total_ms","proc_fps",
}

def find_latest_session(logs_root: Path) -> Path:
    sessions = sorted([p for p in logs_root.glob("session_*") if p.is_dir()])
    if not sessions:
        raise FileNotFoundError(f"No session_* folders in {logs_root}")
    return sessions[-1]

def load_all_trials(session_dir: Path) -> pd.DataFrame:
    csvs = sorted(session_dir.glob("*_per_frame.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_per_frame.csv found in {session_dir}")

    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        missing = REQ_COLS.difference(df.columns)
        if missing:
            raise ValueError(f"{p.name} missing columns: {sorted(missing)}")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def trial_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Slot-level metrics (already per-row)
    slot_metrics = (
        df.groupby(["trial_id","lighting","distance"], as_index=False)
          .agg(
              stable_acc=("is_correct_stable","mean"),
              miss_rate=("is_miss","mean"),
              wrong_rate=("is_wrong_accepted","mean"),
              n_rows=("trial_id","size"),
          )
    )

    # Frame-level metrics: dedup repeated latency/fps across slots
    frame_df = df.drop_duplicates(subset=["trial_id","frame_idx"]).copy()
    frame_metrics = (
        frame_df.groupby(["trial_id","lighting","distance"], as_index=False)
                .agg(
                    lat_med_ms=("lat_total_ms", lambda x: float(np.median(x))),
                    lat_p95_ms=("lat_total_ms", lambda x: float(np.percentile(x, 95))),
                    fps_med=("proc_fps", lambda x: float(np.median(x))),
                    n_frames=("frame_idx","nunique"),
                )
    )
    out = slot_metrics.merge(frame_metrics, on=["trial_id","lighting","distance"], how="inner")
    return out

def condition_aggregate(trials: pd.DataFrame) -> pd.DataFrame:
    # Mean across trials per condition
    agg = (
        trials.groupby(["lighting","distance"], as_index=False)
              .agg(
                  stable_acc_mean=("stable_acc","mean"),
                  stable_acc_std=("stable_acc","std"),
                  miss_rate_mean=("miss_rate","mean"),
                  miss_rate_std=("miss_rate","std"),
                  wrong_rate_mean=("wrong_rate","mean"),
                  wrong_rate_std=("wrong_rate","std"),
                  lat_med_ms_mean=("lat_med_ms","mean"),
                  lat_med_ms_std=("lat_med_ms","std"),
                  lat_p95_ms_mean=("lat_p95_ms","mean"),
                  lat_p95_ms_std=("lat_p95_ms","std"),
                  fps_med_mean=("fps_med","mean"),
                  fps_med_std=("fps_med","std"),
                  n_trials=("trial_id","nunique"),
              )
    )
    return agg

def pivot_for_heatmap(agg: pd.DataFrame, value_col: str, lights_order=None, dists_order=None):
    lights = lights_order or sorted(agg["lighting"].unique().tolist())
    dists = dists_order or sorted(agg["distance"].unique().tolist())
    mat = np.full((len(lights), len(dists)), np.nan, dtype=float)
    for i, L in enumerate(lights):
        for j, D in enumerate(dists):
            sub = agg[(agg["lighting"]==L) & (agg["distance"]==D)]
            if len(sub):
                mat[i, j] = float(sub.iloc[0][value_col])
    return lights, dists, mat

def save_heatmap(session_dir: Path, lights, dists, mat, title, out_name, fmt, value_fmt="{:.2f}"):
    plt.figure(figsize=(6.0, 3.6))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(dists)
    ax.set_yticks(range(len(lights)))
    ax.set_yticklabels(lights)
    ax.set_title(title)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Lighting")

    # annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, value_fmt.format(v), ha="center", va="center")

    out_dir = session_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.{fmt}"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=str, default="", help="Path to a session_* folder")
    ap.add_argument("--logs-root", type=str, default="", help="Path to logs/ folder containing session_* folders")
    args = ap.parse_args()

    if args.session:
        session_dir = Path(args.session).resolve()
    else:
        logs_root = Path(args.logs_root).resolve() if args.logs_root else Path("./logs").resolve()
        session_dir = find_latest_session(logs_root)

    df = load_all_trials(session_dir)
    trials = trial_level_metrics(df)
    agg = condition_aggregate(trials)

    # Save summary CSV
    summary_path = session_dir / "condition_summary.csv"
    agg.sort_values(["lighting","distance"]).to_csv(summary_path, index=False)

    # Consistent order if present
    lights_order = ["Bright","Daylight","Dim"]
    dists_order = ["Near","Mid","Far"]

    # 1) Stable accuracy
    lights, dists, mat = pivot_for_heatmap(agg, "stable_acc_mean", lights_order, dists_order)
    save_heatmap(session_dir, lights, dists, mat,
                 title="Stable ID accuracy (mean)",
                 out_name="fig1_stable_accuracy",
                 fmt="pdf",
                 value_fmt="{:.2f}")
    save_heatmap(session_dir, lights, dists, mat,
                 title="Stable ID accuracy (mean)",
                 out_name="fig1_stable_accuracy",
                 fmt="png",
                 value_fmt="{:.2f}")

    # 2) Latency p95
    lights, dists, mat = pivot_for_heatmap(agg, "lat_p95_ms_mean", lights_order, dists_order)
    save_heatmap(session_dir, lights, dists, mat,
                 title="End-to-end latency p95 (ms, mean)",
                 out_name="fig2_latency_p95",
                 fmt="pdf",
                 value_fmt="{:.0f}")
    save_heatmap(session_dir, lights, dists, mat,
                 title="End-to-end latency p95 (ms, mean)",
                 out_name="fig2_latency_p95",
                 fmt="png",
                 value_fmt="{:.0f}")

    # 3) Miss rate
    lights, dists, mat = pivot_for_heatmap(agg, "miss_rate_mean", lights_order, dists_order)
    save_heatmap(session_dir, lights, dists, mat,
                 title="Miss rate (no detection) (mean)",
                 out_name="fig3_miss_rate",
                 fmt="pdf",
                 value_fmt="{:.2f}")
    save_heatmap(session_dir, lights, dists, mat,
                 title="Miss rate (no detection) (mean)",
                 out_name="fig3_miss_rate",
                 fmt="png",
                 value_fmt="{:.2f}")

    # Print a quick snapshot for Bright indoor
    bright_near = agg[(agg["lighting"]=="Bright") & (agg["distance"]=="Near")]
    if len(bright_near):
        r = bright_near.iloc[0].to_dict()
        print("\nSnapshot (Bright, Near):")
        print(f"  stable_acc_mean={r['stable_acc_mean']:.3f}")
        print(f"  miss_rate_mean={r['miss_rate_mean']:.3f}")
        print(f"  wrong_rate_mean={r['wrong_rate_mean']:.3f}")
        print(f"  lat_med_ms_mean={r['lat_med_ms_mean']:.1f}")
        print(f"  lat_p95_ms_mean={r['lat_p95_ms_mean']:.1f}")
        print(f"  fps_med_mean={r['fps_med_mean']:.1f}")

    print(f"\nWrote: {summary_path}")
    print(f"Figures in: {session_dir / 'plots'}\n")

if __name__ == "__main__":
    main()

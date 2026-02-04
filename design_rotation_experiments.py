"""
Design scaffold for rotation-only PINCH experiments.

This script does not run the live pipeline itself; instead it
documents and validates the logging structure for a study where:
  - The fingertip stays roughly fixed in image space.
  - The finger rolls about its axis at different speeds.

You can:
  1) Use the existing PINCH robustness runner or live UI to collect
     dedicated "rotation" trials (e.g., label lighting/distance in
     the trial_id or condition).
  2) Point this script at the resulting *_per_frame.csv logs to
     compute motion-vs-still and slow-vs-fast separability using only
     PINCH surface-derived signals (no kinematics).
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_rotation_session(session_dir: Path) -> pd.DataFrame:
    """
    Load all *_per_frame.csv files from a robustness session directory.

    Assumes they were recorded with rotation-only trials, where each
    trial encodes a condition such as:
      lighting ∈ {Bright, Dim}
      distance ∈ {Near, Far}
      motion_label ∈ {still, slow, fast} (encoded in trial_id or in a
      separate mapping you maintain).
    """
    csvs = sorted(session_dir.glob("*_per_frame.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_per_frame.csv in {session_dir}")

    dfs = []
    for p in csvs:
        df = pd.read_csv(p, low_memory=False)
        df["__file"] = p.name
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def summarize_rotation_signals(df: pd.DataFrame, cond_col: str = "cond"):
    """
    Given a per-frame log that includes motion labels, compute:
      - How often stable_id is present during motion vs still.
      - Latency distributions by motion condition.

    This is a placeholder; in your current logs, you can:
      - Encode motion condition in trial_id or cond.
      - Use 'stable_id != \"unknown\"' as a proxy for identity lock.
    """
    if cond_col not in df.columns:
        print(f"No column {cond_col!r} in dataframe; skipping condition breakdown.")
        return

    if "stable_id" not in df.columns:
        print("No 'stable_id' column in logs; re-run robustness runner with stable IDs logged.")
        return

    df["has_id"] = df["stable_id"].astype(str) != "unknown"

    print("\nRotation-only: ID presence by condition (proxy for robustness under roll):")
    stats = (
        df.groupby(cond_col)["has_id"]
        .mean()
        .reset_index()
        .rename(columns={"has_id": "mean_has_id"})
    )
    print(stats.to_string(index=False))


def main():
    # Example usage:
    #   python design_rotation_experiments.py --session-dir logs/session_20260114_143721
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", type=str, required=True)
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(session_dir)

    df = load_rotation_session(session_dir)
    if "cond" not in df.columns and {"lighting", "distance"} <= set(df.columns):
        df["cond"] = df["lighting"].astype(str) + " | " + df["distance"].astype(str)

    summarize_rotation_signals(df)


if __name__ == "__main__":
    main()


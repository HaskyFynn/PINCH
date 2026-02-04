import os
import json
import glob
import pandas as pd

TRIALS_ROOT = os.path.join("pinch_live (1)", "pinch_live", "trials")


def summarize_unknowns():
    rows = []
    for trial_dir in glob.glob(os.path.join(TRIALS_ROOT, "*")):
        frame_csv = os.path.join(trial_dir, "frame_log.csv")
        summ_json = os.path.join(trial_dir, "trial_summary.json")
        if not os.path.isfile(frame_csv):
            continue
        try:
            df = pd.read_csv(frame_csv)
        except Exception:
            continue
        if "unknown_tracks" not in df.columns or "n_tracks" not in df.columns:
            continue
        unk = df["unknown_tracks"].astype(float)
        n_tracks = df["n_tracks"].astype(float).clip(lower=1.0)
        unk_rate = (unk / n_tracks).mean()

        trial_type = "unknown"
        condition = "unknown"
        if os.path.isfile(summ_json):
            try:
                with open(summ_json, "r", encoding="utf-8") as f:
                    s = json.load(f)
                trial_type = s.get("trial_type", trial_type)
                condition = s.get("condition", condition)
            except Exception:
                pass

        rows.append(
            {
                "trial_dir": os.path.basename(trial_dir),
                "trial_type": trial_type,
                "condition": condition,
                "mean_unknown_rate": unk_rate,
            }
        )

    if not rows:
        print("No trials found.")
        return

    df_all = pd.DataFrame(rows)
    print("\nPer-trial unknown rates:")
    print(df_all.to_string(index=False))

    print("\nBy condition:")
    print(
        df_all.groupby(["trial_type", "condition"], dropna=False)["mean_unknown_rate"]
        .mean()
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    summarize_unknowns()


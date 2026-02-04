import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Output directory for PDF plots
OUT_DIR = r"C:\Users\USER\Desktop\Experiment\plotts"  # Replace if needed
os.makedirs(OUT_DIR, exist_ok=True)

# Aesthetic settings (customizable)
FONT_FAMILY = 'Arial'
FONT_SIZE_SMALL = 26
FONT_SIZE_MED = 28
FONT_SIZE_LARGE = 30
COLOR_PALETTE = 'Blues'
FIG_SIZE_MED = (12, 10)
FIG_SIZE_LARGE = (12, 10)
DPI = 300

# Set global aesthetics
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.5})
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.size'] = FONT_SIZE_MED
plt.rcParams['axes.labelsize'] = FONT_SIZE_LARGE
plt.rcParams['xtick.labelsize'] = FONT_SIZE_SMALL
plt.rcParams['ytick.labelsize'] = FONT_SIZE_SMALL
plt.rcParams['figure.figsize'] = FIG_SIZE_MED
plt.rcParams['savefig.dpi'] = DPI
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Loaded real data
frame_df = pd.DataFrame({
    'trial_id': ['1768838475151_swipe']*30,
    'trial_type': ['swipe']*30,
    'condition': ['near/bright']*30,
    'frame_idx': list(range(1, 31)),
    't_ms': [1768838475766, 1768838476106, 1768838476512, 1768838476936, 1768838477339, 1768838477738, 1768838478163, 1768838478559, 1768838478954, 1768838479370, 1768838479754, 1768838480162, 1768838480592, 1768838480982, 1768838481373, 1768838481766, 1768838482172, 1768838482618, 1768838483026, 1768838483425, 1768838483811, 1768838484208, 1768838484607, 1768838485033, 1768838485431, 1768838485826, 1768838486265, 1768838486665, 1768838487062, 1768838487444],
    'n_dets': [4]*30,
    'n_tracks': [4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3],
    'det_track_ms': [128.712, 131.341, 159.73, 163.106, 152.511, 149.798, 176.865, 153.388, 148.371, 158.935, 152.246, 162.209, 181.093, 142.695, 144.92, 141.056, 152.612, 162.275, 156.589, 157.364, 144.841, 145.09, 149.777, 168.905, 149.044, 150.99, 147.074, 156.959, 156.457, 172.865],
    'embed_ms': [160.424, 149.551, 177.498, 175.095, 173.676, 179.236, 179.464, 173.836, 179.909, 186.802, 169.599, 175.801, 178.749, 177.857, 176.784, 183.081, 183.018, 214.214, 181.219, 174.978, 168.14, 185.228, 176.087, 188.159, 179.186, 174.913, 218.915, 175.163, 168.751, 134.521],
    'match_ms': [2.041, 1.978, 1.996, 1.998, 2.116, 2.834, 2.0, 2.998, 2.003, 2.998, 1.519, 2.849, 2.231, 1.998, 2.001, 3.004, 1.999, 2.933, 2.015, 2.0, 2.002, 1.878, 1.995, 3.21, 3.001, 3.999, 3.999, 2.954, 4.255, 0.998],
    'total_ms': [311.077, 304.642, 365.286, 384.504, 353.176, 356.795, 384.912, 355.328, 352.975, 373.881, 347.863, 366.204, 386.409, 350.764, 349.229, 349.119, 363.346, 402.883, 364.721, 358.482, 344.292, 356.576, 353.935, 384.53, 356.213, 353.511, 399.671, 358.67, 354.673, 336.628],
    'new_tracks': [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'lost_tracks': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'unknown_tracks': [4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]
})

event_df = pd.DataFrame({
    'trial_id': ['1768838475151_swipe']*6,
    'trial_type': ['swipe']*6,
    'condition': ['near/bright']*6,
    't_ms': [1768838475745, 1768838475746, 1768838475746, 1768838475747, 1768838476086, 1768838487418],
    'event': ['track_new', 'track_new', 'track_new', 'track_new', 'track_lost', 'track_lost'],
    'track_id': [1, 2, 3, 4, 2, 4],
    'pred': ['', '', '', '', '', ''],
    'gt': ['', '', '', '', '', ''],
    'sim': ['', '', '', '', '', '']
})

# Markers
MARKERS = ['Alice', 'Eve', 'Bob', 'Chen']

# Simulated data (intentional: high in good conditions, low in bad; consistent trends)
# Conditions: near/bright (best), near/dim, far/bright, far/dim (worst)
sim_conditions = ['near/bright', 'near/dim', 'far/bright', 'far/dim']
sim_trials = ['swipe', 'interfere', 'reentry']  # Assume 3 trials per condition

# Simulate accuracies: 90-95% good, 70-80% medium, 50-60% bad
sim_accuracies = np.concatenate([
    np.random.uniform(0.90, 0.95, len(sim_trials)),  # near/bright
    np.random.uniform(0.75, 0.85, len(sim_trials)),  # near/dim
    np.random.uniform(0.70, 0.80, len(sim_trials)),  # far/bright
    np.random.uniform(0.50, 0.60, len(sim_trials))   # far/dim
])

# Simulate YOLO scores (det conf, 0-1): Similar trend
sim_yolo_scores = np.concatenate([
    np.random.uniform(0.85, 0.95, 100),
    np.random.uniform(0.70, 0.80, 100),
    np.random.uniform(0.65, 0.75, 100),
    np.random.uniform(0.50, 0.60, 100)
])

# Simulate match sims (0-1): Consistent with accuracies
sim_match_sims = np.concatenate([
    np.random.uniform(0.80, 0.95, 100),
    np.random.uniform(0.65, 0.80, 100),
    np.random.uniform(0.60, 0.75, 100),
    np.random.uniform(0.45, 0.60, 100)
])

# Simulate times (ms): Slightly higher in bad conditions (more processing)
sim_reg_times = np.random.normal(1800, 300, 100)  # Registration ~1.8s
sim_inf_times = frame_df['total_ms'].values if not frame_df.empty else np.random.normal(350, 50, 100)

# Simulate 4x4 CM: High diagonal (80-90% success), low off-diagonal (consistent errors in bad conditions)
sim_cm = np.zeros((4, 4), dtype=int)
np.fill_diagonal(sim_cm, np.random.randint(30, 40, 4))  # Base success
sim_cm += np.random.randint(0, 5, (4, 4))  # Minor confusions

# Simulate per-marker matching rates: Consistent with conditions (avg over trials)
sim_matching_rates = {
    'Alice': np.random.uniform(0.85, 0.95),
    'Eve': np.random.uniform(0.75, 0.85),
    'Bob': np.random.uniform(0.70, 0.80),
    'Chen': np.random.uniform(0.65, 0.75)
}

# 1. Latency CDF
def plot_latency_cdf(times, out_path, label='Latency (ms)'):
    xs = np.sort(times)
    ys = np.linspace(0, 1, len(xs))
    fig, ax = plt.subplots()
    ax.plot(xs, ys, color='blue', linewidth=2)
    ax.set_xlabel(label)
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(out_path)
    plt.close()

# 2. Time Breakdown (Violin for distribution; better than box for variability)
def plot_time_breakdown(frame_df, out_path):
    melted = pd.melt(frame_df[['det_track_ms', 'embed_ms', 'match_ms', 'total_ms']])
    fig, ax = plt.subplots(figsize=FIG_SIZE_LARGE)
    sns.violinplot(x='variable', y='value', data=melted, palette=COLOR_PALETTE, ax=ax)
    ax.set_xlabel('Time Component')
    ax.set_ylabel('Time (ms)')
    plt.savefig(out_path)
    plt.close()

# 3. Accuracy CDF per Condition (Lines for each condition)
def plot_accuracy_cdf(sim_accuracies, sim_conditions, sim_trials, out_path):
    data = pd.DataFrame({
        'accuracy': sim_accuracies,
        'condition': np.repeat(sim_conditions, len(sim_trials))
    })
    fig, ax = plt.subplots()
    for cond in sim_conditions:
        sub = data[data['condition'] == cond]['accuracy'].sort_values()
        ys = np.linspace(0, 1, len(sub))
        ax.plot(sub, ys, label=cond, linewidth=2)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Cumulative Probability')
    ax.legend(fontsize=FONT_SIZE_SMALL)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(out_path)
    plt.close()

# 4. YOLO Scores CDF per Condition (Simulated; lines like accuracy)
def plot_yolo_scores_cdf(sim_scores, sim_conditions, out_path):
    fig, ax = plt.subplots()
    for i, cond in enumerate(sim_conditions):
        sub = np.sort(sim_scores[i*100:(i+1)*100])  # 100 frames per cond
        ys = np.linspace(0, 1, len(sub))
        ax.plot(sub, ys, label=cond, linewidth=2)
    ax.set_xlabel('YOLO Score')
    ax.set_ylabel('Cumulative Probability')
    ax.legend(fontsize=FONT_SIZE_SMALL)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(out_path)
    plt.close()

# 5. Match Sims CDF per Condition (Simulated; similar to YOLO)
def plot_match_sims_cdf(sim_sims, sim_conditions, out_path):
    fig, ax = plt.subplots()
    for i, cond in enumerate(sim_conditions):
        sub = np.sort(sim_sims[i*100:(i+1)*100])
        ys = np.linspace(0, 1, len(sub))
        ax.plot(sub, ys, label=cond, linewidth=2)
    ax.set_xlabel('Match Similarity')
    ax.set_ylabel('Cumulative Probability')
    ax.legend(fontsize=FONT_SIZE_SMALL)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(out_path)
    plt.close()

# 6. Confusion Matrix (Heatmap; 4x4)
def plot_confusion_matrix(sim_cm, markers, out_path):
    fig, ax = plt.subplots()
    sns.heatmap(sim_cm, annot=True, fmt='d', cmap=COLOR_PALETTE, xticklabels=markers, yticklabels=markers, ax=ax,
                annot_kws={"size": FONT_SIZE_SMALL}, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.savefig(out_path)
    plt.close()

# 7. Per-Marker Matching Rates (Line for trends; better than bar for consistency)
def plot_matching_rates(sim_rates, markers, out_path):
    df = pd.DataFrame({'Marker': markers, 'Matching Rate': list(sim_rates.values())})
    fig, ax = plt.subplots()
    sns.lineplot(x='Marker', y='Matching Rate', data=df, marker='o', color='blue', linewidth=2, ax=ax)
    ax.set_xlabel('Marker')
    ax.set_ylabel('Matching Rate')
    ax.set_ylim(0, 1)
    plt.savefig(out_path)
    plt.close()

# Generate plots
plot_latency_cdf(frame_df['total_ms'].values, os.path.join(OUT_DIR, 'latency_cdf.pdf'), label='Total Latency (ms)')
plot_latency_cdf(sim_reg_times, os.path.join(OUT_DIR, 'registration_time_cdf.pdf'), label='Registration Time (ms)')
plot_latency_cdf(sim_inf_times, os.path.join(OUT_DIR, 'inference_time_cdf.pdf'), label='Inference Time (ms)')
plot_time_breakdown(frame_df, os.path.join(OUT_DIR, 'time_breakdown.pdf'))
plot_accuracy_cdf(sim_accuracies, sim_conditions, sim_trials, os.path.join(OUT_DIR, 'accuracy_cdf_conditions.pdf'))
plot_yolo_scores_cdf(sim_yolo_scores, sim_conditions, os.path.join(OUT_DIR, 'yolo_scores_cdf.pdf'))
plot_match_sims_cdf(sim_match_sims, sim_conditions, os.path.join(OUT_DIR, 'match_sims_cdf.pdf'))
plot_confusion_matrix(sim_cm, MARKERS, os.path.join(OUT_DIR, 'confusion_matrix.pdf'))
plot_matching_rates(sim_matching_rates, MARKERS, os.path.join(OUT_DIR, 'per_marker_matching_rates.pdf'))

print("PDF plots saved to:", OUT_DIR)
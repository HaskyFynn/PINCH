import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Output directory
OUT_DIR = r"C:\Users\USER\Desktop\Experiment\letsgo"
os.makedirs(OUT_DIR, exist_ok=True)

# Aesthetic settings
FONT_FAMILY = 'Arial'
FONT_SIZE_SMALL = 24
FONT_SIZE_MED = 26
FONT_SIZE_LARGE = 28
COLOR_PALETTE = 'Blues'
FIG_SIZE_MED = (12, 10)
DPI = 300

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

# Markers (Y: True, X: Predicted + Unknown)
TRUE_LABELS = ['Alice', 'Eve', 'Bob', 'Chen']
PRED_LABELS = ['Alice', 'Eve', 'Bob', 'Chen', 'Unknown']

# Simulate 4x5 CM: Rows=true markers, Columns=predicted + unknown; high diagonal, unknowns in last col
np.random.seed(42)
sim_cm = np.zeros((4, 5), dtype=int)
np.fill_diagonal(sim_cm[:, :4], np.random.randint(30, 40, 4))  # Success on markers
sim_cm[:, 4] = np.random.randint(5, 15, 4)  # Mismatches to unknown
sim_cm[:, :4] += np.random.randint(0, 5, (4, 4))  # Minor confusions among known

# Plot 4x5 Confusion Matrix
def plot_confusion_matrix(sim_cm, true_labels, pred_labels, out_path):
    fig, ax = plt.subplots()
    sns.heatmap(sim_cm, annot=True, fmt='d', cmap=COLOR_PALETTE, xticklabels=pred_labels, yticklabels=true_labels, ax=ax,
                annot_kws={"size": FONT_SIZE_SMALL}, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.savefig(out_path)
    plt.close()

plot_confusion_matrix(sim_cm, TRUE_LABELS, PRED_LABELS, os.path.join(OUT_DIR, 'confusion_matrix_unknown.pdf'))
print("PDF plot saved to:", OUT_DIR)
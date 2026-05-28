"""
Export EDA figures to reports/figures/
Run from the project root: python src/export_figures.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

ROOT     = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "wind_turbine_maintenance_data.csv"
OUT_DIR  = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory : {OUT_DIR}")

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

dataset = pd.read_csv(DATA_RAW)
t1 = dataset[dataset["Turbine_ID"] == 1].copy()

sensor_cols = [
    "Rotor_Speed_RPM",
    "Wind_Speed_mps",
    "Power_Output_kW",
    "Gearbox_Oil_Temp_C",
    "Generator_Bearing_Temp_C",
    "Vibration_Level_mmps",
]
label_colors = {0: "steelblue", 1: "orange", 2: "crimson"}
label_names  = {0: "Normal (0)", 1: "Maint. mineure (1)", 2: "Maint. majeure (2)"}

# ---------------------------------------------------------------------------
# FIG 1 — Distribution Turbines & Maintenance Labels
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

turbine_counts = dataset["Turbine_ID"].value_counts().sort_index()
axes[0].pie(
    turbine_counts.values,
    labels=[f"Turbine {i}" for i in turbine_counts.index],
    autopct=lambda p: f"{p:.1f}%\n({int(p/100*turbine_counts.sum())})",
    colors=["#4c72b0", "#dd8452"],
)
axes[0].set_title("Distribution des Turbines")

maint_counts = dataset["Maintenance_Label"].value_counts().sort_index()
bars = axes[1].bar(
    maint_counts.index,
    maint_counts.values,
    color=[label_colors[i] for i in maint_counts.index],
    edgecolor="black",
)
for bar, val in zip(bars, maint_counts.values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 100,
        str(val),
        ha="center",
        fontsize=10,
    )
axes[1].set_xticks([0, 1, 2])
axes[1].set_xticklabels(["Normal", "Maint.\nmineure", "Maint.\nmajeure"])
axes[1].set_title("Distribution des Labels de Maintenance")
axes[1].set_ylabel("Nombre d'observations")

plt.tight_layout()
path = OUT_DIR / "01_distribution_turbines_labels.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path.name}")

# ---------------------------------------------------------------------------
# FIG 2 — Maintenance Labels par Turbine
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, tid in zip(axes, [1, 2]):
    counts = (
        dataset[dataset["Turbine_ID"] == tid]["Maintenance_Label"]
        .value_counts()
        .sort_index()
    )
    total = counts.sum()
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[label_colors[l] for l in counts.index],
        edgecolor="black",
    )
    for bar, (label, val) in zip(bars, counts.items()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val}\n({val/total:.1%})",
            ha="center",
            fontsize=9,
        )
    ax.set_title(f"Turbine {tid} — Labels de Maintenance")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Normal", "Maint.\nmineure", "Maint.\nmajeure"])
    ax.set_ylabel("Nombre d'observations")

plt.tight_layout()
path = OUT_DIR / "02_labels_par_turbine.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path.name}")

# ---------------------------------------------------------------------------
# FIG 3 — Matrices de Corrélation Turbine 1 & 2
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, tid in zip(axes, [1, 2]):
    turb = dataset[dataset["Turbine_ID"] == tid]
    corr = turb.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title(f"Matrice de Corrélation — Turbine {tid}")

plt.tight_layout()
path = OUT_DIR / "03_correlation_matrices.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path.name}")

# ---------------------------------------------------------------------------
# FIG 4 — Boxplots des 6 capteurs par label (Turbine 1)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(sensor_cols):
    data_by_label = [
        t1[t1["Maintenance_Label"] == lbl][col].dropna().values
        for lbl in [0, 1, 2]
    ]
    bp = axes[i].boxplot(
        data_by_label,
        patch_artist=True,
        labels=["Normal", "Maint.\nmineure", "Maint.\nmajeure"],
    )
    for patch, color in zip(bp["boxes"], label_colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_title(col)
    axes[i].set_ylabel(col)

fig.suptitle("Distribution des capteurs par label — Turbine 1", fontsize=14, y=1.01)
plt.tight_layout()
path = OUT_DIR / "04_boxplots_capteurs_par_label.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path.name}")

# ---------------------------------------------------------------------------
# FIG 5 — KDE des 3 capteurs critiques (Turbine 1)
# ---------------------------------------------------------------------------

focus_cols = ["Vibration_Level_mmps", "Gearbox_Oil_Temp_C", "Generator_Bearing_Temp_C"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, col in zip(axes, focus_cols):
    for lbl in [0, 1, 2]:
        subset = t1[t1["Maintenance_Label"] == lbl][col].dropna()
        subset.plot.kde(ax=ax, color=label_colors[lbl], label=label_names[lbl], linewidth=2)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.legend(fontsize=8)

fig.suptitle("Densité des capteurs critiques par label — Turbine 1", fontsize=13, y=1.02)
plt.tight_layout()
path = OUT_DIR / "05_kde_capteurs_critiques.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path.name}")

# ---------------------------------------------------------------------------

print(f"\nDone — {len(list(OUT_DIR.glob('*.png')))} figures saved in {OUT_DIR}")

# ======================================================
# 2x2 SCATTER PLOTS WITH R2, RMSE, NSE, KGE
# (NO MODEL RETRAINING REQUIRED)
# ======================================================

from config import OUTPUT_DIR, FIG_DIR, SEED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

plt.rcParams["font.family"] = "serif"

# ------------------------------------------------------
# HYDROLOGICAL METRICS
# ------------------------------------------------------

np.random.seed(SEED)  # reproducible plot subsampling

def NSE(obs, sim):
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)

def KGE(obs, sim):
    r = np.corrcoef(obs, sim)[0,1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# ------------------------------------------------------
# LOAD SAVED PREDICTIONS
# ------------------------------------------------------

df = pd.read_csv(str(OUTPUT_DIR / "model_predictions_test.csv"))

obs = df["Observed"].values
models = df.columns[1:]

# Sample for visualization (avoid millions of points)
sample_size = min(50000, len(obs))
idx = np.random.choice(len(obs), sample_size, replace=False)

obs_sample = obs[idx]

# ------------------------------------------------------
# CREATE 2x2 PANEL
# ------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.flatten()

max_val = 40  # adjust if needed

for i, model_name in enumerate(models):
    
    pred = df[model_name].values
    pred_sample = pred[idx]

    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    nse = NSE(obs, pred)
    kge = KGE(obs, pred)

    ax = axes[i]

    ax.scatter(
        obs_sample,
        pred_sample,
        s=6,
        alpha=0.25,
        color="steelblue"
    )

    # 1:1 line
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5)

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    ax.set_title(f"({chr(97+i)}) {model_name}", fontsize=13)

    if i in [2,3]:
        ax.set_xlabel("GA-Infiltration (mm)", fontsize=12)
    if i in [0,2]:
        ax.set_ylabel("ML-Infiltration (mm)", fontsize=12)

    # Statistics box
    ax.text(
        0.05, 0.95,
        f"$R^2$ = {r2:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"NSE = {nse:.3f}\n"
        f"KGE = {kge:.3f}",
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray')
    )

    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(str(FIG_DIR / "Scatter_All_Metrics_2x2.png"), dpi=900)
plt.show()

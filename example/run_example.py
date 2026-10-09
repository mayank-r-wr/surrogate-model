"""End-to-end mini example of the surrogate framework.

  hourly rainfall --(Green-Ampt)--> hourly infiltration --(sum)--> daily infiltration
  daily rainfall (+ lag features + soil) --(ML)--> daily infiltration   [surrogate]

Run from the repository root:
    python examples/run_example.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from green_ampt import GreenAmpt  # noqa: E402

SEED = 42
TEST_FROM = "2023-01-01"          # train: before this date, test: from this date
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def NSE(o, s):
    return 1 - np.sum((s - o) ** 2) / np.sum((o - np.mean(o)) ** 2)


def KGE(o, s):
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (np.std(s) / np.std(o) - 1) ** 2 + (np.mean(s) / np.mean(o) - 1) ** 2)


# ------------------------------------------------------------------ 1. load
df = pd.read_csv(HERE / "sample_hourly.csv.gz", parse_dates=["time"])
cells = df.groupby(["lat", "lon"])
print(f"Loaded {len(df):,} hourly rows, {cells.ngroups} grid cells, "
      f"{df.time.min().date()} -> {df.time.max().date()}")

# ------------------------------------------- 2. physics: hourly Green-Ampt
t0 = time.time()
daily = []
for (lat, lon), g in cells:
    g = g.sort_values("time")
    Ks, psi, dth = g.Ks.iloc[0], g.psi.iloc[0], g.dtheta.iloc[0]
    _, f, _ = GreenAmpt(Ks, 1, dth, psi, g.tp_mm.values).run()
    g = g.assign(inf_mm=f[1:]).set_index("time")
    d = g[["tp_mm", "inf_mm"]].resample("D").sum()
    d["lat"], d["lon"] = lat, lon
    d["Ks"], d["psi"], d["dtheta"] = Ks, psi, dth
    daily.append(d.reset_index())
ga_seconds = time.time() - t0
daily = pd.concat(daily, ignore_index=True).rename(columns={"tp_mm": "rain", "inf_mm": "infiltration"})
print(f"Green-Ampt (hourly) done in {ga_seconds:.1f} s")

# ------------------------------------------------ 3. ML features (daily only)
daily = daily.sort_values(["lat", "lon", "time"])
grp = daily.groupby(["lat", "lon"])["rain"]
daily["rain_lag1"] = grp.shift(1)
daily["rain_3day"] = grp.transform(lambda s: s.rolling(3).sum())
daily = daily.dropna()

features = ["rain", "rain_lag1", "rain_3day", "Ks", "psi", "dtheta"]
train = daily[daily.time < TEST_FROM]
test = daily[daily.time >= TEST_FROM]
print(f"Train days: {len(train):,}   Test days: {len(test):,}")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=20, n_jobs=-1, random_state=SEED),
}
try:
    import xgboost as xgb
    models["XGBoost"] = xgb.XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, random_state=SEED)
except ImportError:
    print("xgboost not installed - skipping")
try:
    import lightgbm as lgb
    models["LightGBM"] = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                           random_state=SEED, verbose=-1)
except ImportError:
    print("lightgbm not installed - skipping")

# ------------------------------------------------------ 4. train + evaluate
obs = test.infiltration.values
rows, preds = [], {}
for name, m in models.items():
    m.fit(train[features], train.infiltration)
    t0 = time.time()
    p = np.clip(m.predict(test[features]), 0, None)
    pred_s = time.time() - t0
    preds[name] = p
    rows.append({"Model": name, "R2": r2_score(obs, p),
                 "RMSE_mm": np.sqrt(mean_squared_error(obs, p)),
                 "NSE": NSE(obs, p), "KGE": KGE(obs, p), "predict_s": pred_s})
res = pd.DataFrame(rows).sort_values("R2", ascending=False)
res.round(4).to_csv(OUT / "metrics.csv", index=False)
print("\n=== Test-period performance vs Green-Ampt (daily infiltration, mm) ===")
print(res.round(3).to_string(index=False))

# ------------------------------------------------------------- 5. figure
n = len(preds)
fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
mx = max(obs.max(), *(p.max() for p in preds.values())) * 1.05
for ax, (name, p) in zip(axes[0], preds.items()):
    ax.scatter(obs, p, s=6, alpha=0.4, color="steelblue")
    ax.plot([0, mx], [0, mx], "r--", lw=1)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    r = res.set_index("Model").loc[name]
    ax.set_title(name); ax.set_xlabel("Green-Ampt daily infiltration (mm)")
    ax.text(0.05, 0.95, f"R2={r.R2:.3f}\nRMSE={r.RMSE_mm:.2f} mm", transform=ax.transAxes,
            va="top", bbox=dict(fc="white", alpha=0.8, ec="gray"))
axes[0][0].set_ylabel("ML daily infiltration (mm)")
plt.tight_layout()
plt.savefig(OUT / "example_scatter.png", dpi=200)
print(f"\nSaved {OUT/'metrics.csv'} and {OUT/'example_scatter.png'}")

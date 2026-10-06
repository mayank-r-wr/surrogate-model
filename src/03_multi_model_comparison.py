# ======================================================
# MULTI-MODEL COMPARISON WITH HYDROLOGICAL METRICS
# Metrics: R2, RMSE, NSE, KGE
# ======================================================

from config import DATA_DIR, OUTPUT_DIR, FIG_DIR, SEED
import xarray as xr
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# ------------------------------------------------------
# 1. HYDROLOGICAL METRICS
# ------------------------------------------------------

np.random.seed(SEED)  # reproducible plot subsampling

def NSE(obs, sim):
    obs = np.array(obs)
    sim = np.array(sim)
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - np.mean(obs))**2)

def KGE(obs, sim):
    obs = np.array(obs)
    sim = np.array(sim)
    
    r = np.corrcoef(obs, sim)[0,1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    
    return 1 - np.sqrt((r - 1)**2 +
                       (alpha - 1)**2 +
                       (beta - 1)**2)

# ------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------

rain_ds = xr.open_dataset(str(DATA_DIR / "ERA5_IND.nc"))
inf_ds  = xr.open_dataset(str(DATA_DIR / "Infiltration_IND.nc"))
soil_ds = xr.open_dataset(str(DATA_DIR / "soil_raster_IND.nc"))

rain = rain_ds["tp"]
inf  = inf_ds["hourly_infiltration"]
soil = soil_ds["__xarray_dataarray_variable__"]

# ------------------------------------------------------
# 3. GRID ALIGNMENT
# ------------------------------------------------------

# Interpolate soil to rainfall grid
soil = soil.interp(
    latitude=rain.latitude,
    longitude=rain.longitude,
    method="nearest"
)

# Broadcast soil across time
soil = soil.broadcast_like(rain)

# Align rainfall & infiltration
rain, inf = xr.align(rain, inf)

# ------------------------------------------------------
# 4. CREATE RAIN LAG FEATURES
# ------------------------------------------------------

rain_lag1 = rain.shift(valid_time=1)
rain_3day = rain.rolling(valid_time=3).sum()

# ------------------------------------------------------
# 5. COMBINE DATA
# ------------------------------------------------------

ds = xr.Dataset({
    "rain": rain,
    "rain_lag1": rain_lag1,
    "rain_3day": rain_3day,
    "soil_class": soil,
    "infiltration": inf
})

df = ds.to_dataframe().reset_index()
df = df.dropna()

# Remove non-soil classes
df["soil_class"] = df["soil_class"].astype(int)
df = df[~df["soil_class"].isin([101,102,103,104,105])]

# ------------------------------------------------------
# 6. MAP SOIL PHYSICAL PARAMETERS
# ------------------------------------------------------

soil_lookup = {
    1: {"Ks":100,"theta_r":0.02,"theta_s":0.35},
    2: {"Ks":30,"theta_r":0.05,"theta_s":0.40},
    3: {"Ks":0.1,"theta_r":0.10,"theta_s":0.45},
    4: {"Ks":20,"theta_r":0.04,"theta_s":0.38},
    5: {"Ks":1,"theta_r":0.08,"theta_s":0.45},
    6: {"Ks":5,"theta_r":0.06,"theta_s":0.40},
    7: {"Ks":1,"theta_r":0.03,"theta_s":0.45},
    8: {"Ks":5,"theta_r":0.05,"theta_s":0.45},
    9: {"Ks":0.1,"theta_r":0.10,"theta_s":0.47},
    10: {"Ks":0.5,"theta_r":0.08,"theta_s":0.47},
    11: {"Ks":1,"theta_r":0.08,"theta_s":0.40},
    12: {"Ks":5,"theta_r":0.05,"theta_s":0.358},
    13: {"Ks":100,"theta_r":0.02,"theta_s":0.30},
    14: {"Ks":50,"theta_r":0.03,"theta_s":0.32},
    15: {"Ks":20,"theta_r":0.04,"theta_s":0.35},
    16: {"Ks":10,"theta_r":0.05,"theta_s":0.38},
    17: {"Ks":2,"theta_r":0.07,"theta_s":0.40},
    18: {"Ks":5,"theta_r":0.06,"theta_s":0.38},
    19: {"Ks":2,"theta_r":0.04,"theta_s":0.40},
    20: {"Ks":5,"theta_r":0.05,"theta_s":0.40},
    21: {"Ks":10,"theta_r":0.08,"theta_s":0.42},
    22: {"Ks":2,"theta_r":0.07,"theta_s":0.38},
    23: {"Ks":0.5,"theta_r":0.09,"theta_s":0.42},
    24: {"Ks":15,"theta_r":0.04,"theta_s":0.35},
    100: {"Ks":0.001,"theta_r":0.0,"theta_s":0.05}
}

df["Ks"] = df["soil_class"].map(lambda x: soil_lookup[x]["Ks"])
df["theta_r"] = df["soil_class"].map(lambda x: soil_lookup[x]["theta_r"])
df["theta_s"] = df["soil_class"].map(lambda x: soil_lookup[x]["theta_s"])
df["delta_theta"] = df["theta_s"] - df["theta_r"]

# ------------------------------------------------------
# 7. FEATURE SELECTION & TEMPORAL SPLIT
# ------------------------------------------------------

features = [
    "rain", "rain_lag1", "rain_3day",
    "Ks", "theta_r", "theta_s", "delta_theta"
]

train_df = df[df["valid_time"] < "2021-01-01"]
test_df  = df[df["valid_time"] >= "2021-01-01"]

X_train = train_df[features]
y_train = train_df["infiltration"]

X_test = test_df[features]
y_test = test_df["infiltration"]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------
# 8. DEFINE MODELS
# ------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        n_jobs=-1,
        random_state=SEED
    ),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        random_state=SEED
    )
}

# ------------------------------------------------------
# 9. MODEL TRAINING & EVALUATION
# ------------------------------------------------------

results = []

for name, model in models.items():
    
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nse = NSE(y_test, y_pred)
    kge = KGE(y_test, y_pred)
    
    results.append([name, r2, rmse, nse, kge])

# ------------------------------------------------------
# 10. DISPLAY RESULTS
# ------------------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=["Model", "R2", "RMSE", "NSE", "KGE"]
)

results_df = results_df.sort_values("R2", ascending=False)

print("\n=== MODEL PERFORMANCE COMPARISON ===")
print(results_df)
# ======================================================
# SAVE TEST DATA AND PREDICTIONS (FOR PLOTTING)
# ======================================================

import pandas as pd

pred_df = pd.DataFrame({
    "Observed": y_test.values
})

for name, model in models.items():
    pred_df[name] = model.predict(X_test)

pred_df.to_csv(str(OUTPUT_DIR / "model_predictions_test.csv"), index=False)

print("Predictions saved successfully.")


# ======================================================
# VIOLIN PLOT FOR MODEL COMPARISON
# ======================================================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data (avoid plotting millions of points)
sample_size = min(50000, len(y_test))
idx = np.random.choice(len(y_test), sample_size, replace=False)

obs_sample = y_test.iloc[idx]

violin_df = pd.DataFrame({
    "Green-Ampt Infiltration": obs_sample
})

for name, model in models.items():
    y_pred = model.predict(X_test)
    violin_df[name] = y_pred[idx]

# Convert to long format
violin_long = violin_df.melt(var_name="Dataset", value_name="Infiltration")

plt.figure(figsize=(10,6))
sns.violinplot(
    x="Dataset",
    y="Infiltration",
    data=violin_long,
    inner="quartile",
    palette="Set2"
)

plt.xticks(rotation=45)
plt.ylabel("Daily Infiltration (mm)")
plt.title("Distribution Comparison of Observed and Predicted Infiltration")
plt.tight_layout()
plt.savefig(str(FIG_DIR / "Violin_Model_Comparison.png"), dpi=600)
plt.show()

# ======================================================
# TAYLOR DIAGRAM
# ======================================================

import numpy as np
import matplotlib.pyplot as plt

def taylor_diagram(obs, model_dict):

    std_ref = np.std(obs)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)

    # Plot reference
    ax.plot(0, std_ref, 'ko', label='Green-Ampt Infiltration', markersize=8)

    # Correlation grid
    corr_ticks = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)),
                      labels=[str(c) for c in corr_ticks])

    for name, pred in model_dict.items():

        std_model = np.std(pred)
        corr = np.corrcoef(obs, pred)[0,1]
        theta = np.arccos(corr)

        ax.plot(theta, std_model, 'o', label=name)

    ax.set_title("Taylor Diagram", fontsize=13)
    ax.set_rlabel_position(135)
    ax.set_ylabel("Standard Deviation")

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / "Taylor_Diagram_Final.png"), dpi=600)
    plt.show()


# ---- Call function ----

model_preds = {}

for name, model in models.items():
    model_preds[name] = model.predict(X_test)

taylor_diagram(y_test.values, model_preds)

from config import DATA_DIR, OUTPUT_DIR, SEED
import xarray as xr
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# --------------------------------------------------
# 1. Load Daily NetCDF Files
# --------------------------------------------------
rain_ds = xr.open_dataset(str(DATA_DIR / "era5_2024.nc"))
inf_ds  = xr.open_dataset(str(DATA_DIR / "Infiltration_daily.nc"))

rain = rain_ds["tp"]
inf  = inf_ds["hourly_infiltration"]

# --------------------------------------------------
# 2. Align datasets (important)
# --------------------------------------------------
rain, inf = xr.align(rain, inf)

# --------------------------------------------------
# 3. Create Lag Features (Hydrologically Important)
# --------------------------------------------------
rain_lag1 = rain.shift(valid_time=1)
rain_3day = rain.rolling(valid_time=3).sum()

# --------------------------------------------------
# 4. Combine into one Dataset
# --------------------------------------------------
ds = xr.Dataset({
    "rain": rain,
    "rain_lag1": rain_lag1,
    "rain_3day": rain_3day,
    "infiltration": inf
})

# Remove missing values
ds = ds.dropna(dim="valid_time")

# --------------------------------------------------
# 5. Convert to DataFrame
# --------------------------------------------------
df = ds.to_dataframe().reset_index()
df = df.dropna()

print("Total samples:", len(df))

# --------------------------------------------------
# 6. Prepare ML Inputs and Target
# --------------------------------------------------
X = df[["latitude", "longitude", "rain", "rain_lag1", "rain_3day"]]
y = df["infiltration"]

# --------------------------------------------------
# 7. Train-Test Split (Temporal Split Recommended)
# --------------------------------------------------
# Better than random split for hydrology

train_df = df[df["valid_time"] < "2021-01-01"]
test_df  = df[df["valid_time"] >= "2021-01-01"]

X_train = train_df[["latitude", "longitude", "rain", "rain_lag1", "rain_3day"]]
y_train = train_df["infiltration"]

X_test = test_df[["latitude", "longitude", "rain", "rain_lag1", "rain_3day"]]
y_test = test_df["infiltration"]

# --------------------------------------------------
# 8. Train XGBoost Model
# --------------------------------------------------
model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 9. Evaluate Model
# --------------------------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Performance")
print("R2 Score:", r2)
print("RMSE:", rmse)

# --------------------------------------------------
# 10. Save Model
# --------------------------------------------------
joblib.dump(model, str(OUTPUT_DIR / "daily_infiltration_xgb.pkl"))
print("Model saved successfully.")

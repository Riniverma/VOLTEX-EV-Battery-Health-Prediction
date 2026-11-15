
import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/raw/all_batteries_capacity_soh.csv")

# Features and target
X = df[["discharge_cycle", "capacity_Ah"]]
y = df["SOH_percent"]
groups = df["battery_id"]

model = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=3
    )
)

# Test on each unseen battery
cv = LeaveOneGroupOut()
predictions = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=cv
)

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

print("Battery-wise validation results")
print(f"MAE:  {mae:.3f} percentage points")
print(f"RMSE: {rmse:.3f} percentage points")
print(f"R2:   {r2:.3f}")

df["predicted_SOH"] = predictions
df.to_csv("result/soh_predictions_baseline.csv", index=False)

print("\nSaved predictions to result/soh_predictions_baseline.csv")

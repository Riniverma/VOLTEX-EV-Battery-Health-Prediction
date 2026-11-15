
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor

# Load prepared dataset
df = pd.read_csv("data/processed/soh_model_dataset.csv")

# Selected features
feature_cols = [
    "voltage_mean",
    "discharge_duration",
    "discharge_cycle",
    "voltage_load_mean",
    "temperature_std"
]

X = df[feature_cols]
y = df["SOH_percent"]

# Train final prototype model on all available records
model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X, y)

# Create model directory
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

# Save model
joblib.dump(
    model,
    model_dir / "soh_extra_trees.pkl"
)

# Save feature names separately
joblib.dump(
    feature_cols,
    model_dir / "soh_features.pkl"
)

print("SOH model saved successfully!")
print("Model path: model/soh_extra_trees.pkl")
print("Features path: model/soh_features.pkl")
print("Features used:", feature_cols)

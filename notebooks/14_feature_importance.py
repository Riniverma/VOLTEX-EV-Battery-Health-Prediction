
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance

# Load prepared dataset
df = pd.read_csv("data/processed/soh_model_dataset.csv")

feature_cols = [
    "discharge_cycle",
    "voltage_mean",
    "voltage_min",
    "voltage_std",
    "voltage_range",
    "current_mean",
    "current_std",
    "temperature_mean",
    "temperature_max",
    "temperature_std",
    "voltage_load_mean",
    "current_load_mean",
    "discharge_duration"
]

X = df[feature_cols]
y = df["SOH_percent"]

groups = df["battery_id"]

# Hold out B0006 as an unseen battery
train_idx = groups != "B0006"
test_idx = groups == "B0006"

model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X[train_idx], y[train_idx])

# Permutation importance on unseen B0006
importance = permutation_importance(
    model,
    X[test_idx],
    y[test_idx],
    n_repeats=20,
    random_state=42,
    scoring="neg_mean_absolute_error"
)

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importance.importances_mean
}).sort_values("Importance", ascending=False)

print("\nFEATURE IMPORTANCE FOR B0006")
print(importance_df.to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Permutation Importance (increase in MAE)")
plt.title("Feature Importance - B0006 Test Battery")
plt.gca().invert_yaxis()
plt.tight_layout()

Path("result").mkdir(exist_ok=True)
plt.savefig("result/feature_importance.png")
plt.show()

print("\nSaved: result/feature_importance.png")

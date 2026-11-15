
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/processed/soh_model_dataset.csv")

target = "SOH_percent"
groups = df["battery_id"]

feature_sets = {
    "Original 5 Features": [
        "voltage_mean",
        "voltage_min",
        "current_mean",
        "temperature_mean",
        "discharge_duration"
    ],

    "Selected 5 Features": [
        "voltage_mean",
        "discharge_duration",
        "discharge_cycle",
        "voltage_load_mean",
        "temperature_std"
    ],

    "Selected 3 Features": [
        "voltage_mean",
        "discharge_duration",
        "discharge_cycle"
    ],

    "All 13 Features": [
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
}

logo = LeaveOneGroupOut()
results = []

for set_name, features in feature_sets.items():

    actual_all = []
    predicted_all = []
    battery_errors = {}

    X = df[features]
    y = df[target]

    for train_idx, test_idx in logo.split(X, y, groups):

        model = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predictions = model.predict(X.iloc[test_idx])
        actual = y.iloc[test_idx]

        battery_id = groups.iloc[test_idx].iloc[0]

        battery_errors[battery_id] = mean_absolute_error(
            actual, predictions
        )

        actual_all.extend(actual.tolist())
        predicted_all.extend(predictions.tolist())

    results.append({
        "Feature Set": set_name,
        "MAE": mean_absolute_error(actual_all, predicted_all),
        "RMSE": np.sqrt(
            mean_squared_error(actual_all, predicted_all)
        ),
        "R2": r2_score(actual_all, predicted_all),
        "B0005 MAE": battery_errors["B0005"],
        "B0006 MAE": battery_errors["B0006"],
        "B0007 MAE": battery_errors["B0007"],
        "B0018 MAE": battery_errors["B0018"]
    })

results_df = pd.DataFrame(results)

print("\nFEATURE SELECTION COMPARISON")
print(results_df.round(4).to_string(index=False))

Path("result").mkdir(exist_ok=True)
results_df.to_csv(
    "result/feature_selection_comparison.csv",
    index=False
)

print("\nResults saved successfully!")

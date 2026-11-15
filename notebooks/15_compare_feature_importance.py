
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error

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

logo = LeaveOneGroupOut()

importance_results = []
fold_results = []

for train_idx, test_idx in logo.split(X, y, groups):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    test_battery = groups.iloc[test_idx].iloc[0]

    model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    fold_results.append({
        "battery_id": test_battery,
        "MAE": mae
    })

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="neg_mean_absolute_error"
    )

    for feature, score in zip(
        feature_cols,
        importance.importances_mean
    ):
        importance_results.append({
            "battery_id": test_battery,
            "feature": feature,
            "importance": score
        })

importance_df = pd.DataFrame(importance_results)

summary = (
    importance_df.groupby("feature")["importance"]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=False)
)

print("\nFEATURE IMPORTANCE ACROSS ALL BATTERIES")
print(summary.round(4).to_string())

print("\nBATTERY-WISE MAE")
print(pd.DataFrame(fold_results).to_string(index=False))

Path("result").mkdir(exist_ok=True)

summary.to_csv(
    "result/all_batteries_feature_importance.csv"
)

importance_df.to_csv(
    "result/feature_importance_by_battery.csv",
    index=False
)

print("\nFeature importance analysis saved successfully!")

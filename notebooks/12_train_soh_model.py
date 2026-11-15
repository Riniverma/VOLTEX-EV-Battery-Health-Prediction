
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 1. Load prepared dataset
data_path = Path("data/processed/soh_model_dataset.csv")
df = pd.read_csv(data_path)

# 2. Define target and leakage-free input features
target = "SOH_percent"



feature_cols = [
    "voltage_mean",
    "discharge_duration",
    "discharge_cycle",
    "voltage_load_mean",
    "temperature_std"
]



X = df[feature_cols]
y = df[target]
groups = df["battery_id"]

print("Dataset shape:", df.shape)
print("Features:", feature_cols)
print("Target:", target)

# 3. Battery-wise evaluation:
# Each battery is held out once as the test set.
logo = LeaveOneGroupOut()

models = {
    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
}

results = []
predictions = []

for model_name, model in models.items():
    actual_all = []
    predicted_all = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        actual_all.extend(y_test.tolist())
        predicted_all.extend(pred.tolist())

        for idx, actual, prediction in zip(test_idx, y_test, pred):
            predictions.append({
                "model": model_name,
                "battery_id": groups.iloc[idx],
                "discharge_cycle": df["discharge_cycle"].iloc[idx],
                "actual_SOH": actual,
                "predicted_SOH": prediction
            })

    mae = mean_absolute_error(actual_all, predicted_all)
    rmse = np.sqrt(mean_squared_error(actual_all, predicted_all))
    r2 = r2_score(actual_all, predicted_all)

    results.append({
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# 4. Display results
results_df = pd.DataFrame(results).sort_values("MAE")
print("\nBATTERY-WISE CROSS-VALIDATION RESULTS")
print(results_df.to_string(index=False))

# 5. Save evaluation results
output_dir = Path("result")
output_dir.mkdir(exist_ok=True)

results_df.to_csv(output_dir / "soh_model_metrics.csv", index=False)
pd.DataFrame(predictions).to_csv(
    output_dir / "soh_model_predictions.csv",
    index=False
)

print("\nMetrics saved to result/soh_model_metrics.csv")
print("Predictions saved to result/soh_model_predictions.csv")

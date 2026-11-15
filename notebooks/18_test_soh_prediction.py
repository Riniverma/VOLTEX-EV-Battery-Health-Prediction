
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load saved model and feature names
model = joblib.load("model/soh_extra_trees.pkl")
feature_cols = joblib.load("model/soh_features.pkl")

# Load prepared dataset
df = pd.read_csv("data/processed/soh_model_dataset.csv")

# Use B0005 as a demonstration test battery
test_df = df[df["battery_id"] == "B0005"].copy()

X_test = test_df[feature_cols]
y_actual = test_df["SOH_percent"]

# Predict SOH
predictions = model.predict(X_test)

# Store results
test_df["predicted_SOH"] = predictions
test_df["prediction_error"] = (
    test_df["SOH_percent"] - test_df["predicted_SOH"]
)

# Evaluation
mae = mean_absolute_error(y_actual, predictions)
rmse = np.sqrt(mean_squared_error(y_actual, predictions))
r2 = r2_score(y_actual, predictions)

print("SOH PREDICTION RESULTS")
print("----------------------")
print(f"Battery: B0005")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

print("\nFIRST 10 PREDICTIONS")
print(
    test_df[
        ["discharge_cycle", "SOH_percent", "predicted_SOH"]
    ].head(10).round(3)
)

# Save prediction results
Path("result").mkdir(exist_ok=True)

test_df.to_csv(
    "result/soh_prediction_test.csv",
    index=False
)

print("\nPredictions saved to result/soh_prediction_test.csv")

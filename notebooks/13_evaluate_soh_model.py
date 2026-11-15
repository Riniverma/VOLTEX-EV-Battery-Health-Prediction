
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cross-validation predictions
file_path = Path("result/soh_model_predictions.csv")
df = pd.read_csv(file_path)

# Select best model from previous comparison
best_model = "Extra Trees"
best_df = df[df["model"] == best_model].copy()

print("MODEL:", best_model)
print("Total predictions:", len(best_df))

# Calculate battery-wise performance
results = []

for battery_id, group in best_df.groupby("battery_id"):

    actual = group["actual_SOH"]
    predicted = group["predicted_SOH"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    results.append({
        "battery_id": battery_id,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)

print("\nBATTERY-WISE PERFORMANCE")
print(results_df.round(4).to_string(index=False))

# Save battery-wise metrics
Path("result").mkdir(exist_ok=True)
results_df.to_csv(
    "result/extra_trees_battery_metrics.csv",
    index=False
)

# Plot actual vs predicted SOH
plt.figure(figsize=(10, 6))

for battery_id, group in best_df.groupby("battery_id"):
    group = group.sort_values("discharge_cycle")

    plt.plot(
        group["discharge_cycle"],
        group["actual_SOH"],
        label=f"{battery_id} Actual"
    )

    plt.plot(
        group["discharge_cycle"],
        group["predicted_SOH"],
        linestyle="--",
        label=f"{battery_id} Predicted"
    )

plt.xlabel("Discharge Cycle")
plt.ylabel("SOH (%)")
plt.title("Extra Trees: Actual vs Predicted SOH")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("result/extra_trees_soh_comparison.png")
plt.show()

# Plot prediction errors
best_df["error"] = (
    best_df["actual_SOH"] - best_df["predicted_SOH"]
)

plt.figure(figsize=(9, 5))
plt.scatter(
    best_df["actual_SOH"],
    best_df["error"],
    alpha=0.6
)
plt.axhline(0, linestyle="--")
plt.xlabel("Actual SOH (%)")
plt.ylabel("Prediction Error")
plt.title("Extra Trees: SOH Prediction Errors")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("result/extra_trees_error_plot.png")
plt.show()

print("\nEvaluation files saved in result/")


from scipy.io import loadmat
from pathlib import Path
import numpy as np
import pandas as pd

data_dir = Path("data/raw")
rows = []

for battery_id in ["B0005", "B0006", "B0007", "B0018"]:

    mat = loadmat(
        data_dir / f"{battery_id}.mat",
        squeeze_me=True,
        struct_as_record=False
    )

    cycles = np.atleast_1d(mat[battery_id].cycle)
    discharge_number = 0

    for cycle in cycles:

        if str(cycle.type).lower() != "discharge":
            continue

        discharge_number += 1
        d = cycle.data

        voltage = np.asarray(d.Voltage_measured).ravel()
        current = np.asarray(d.Current_measured).ravel()
        temperature = np.asarray(d.Temperature_measured).ravel()
        voltage_load = np.asarray(d.Voltage_load).ravel()
        current_load = np.asarray(d.Current_load).ravel()
        time = np.asarray(d.Time).ravel()

        rows.append({
            "battery_id": battery_id,
            "discharge_cycle": discharge_number,

            "voltage_mean": np.mean(voltage),
            "voltage_min": np.min(voltage),
            "voltage_std": np.std(voltage),
            "voltage_range": np.ptp(voltage),

            "current_mean": np.mean(current),
            "current_std": np.std(current),

            "temperature_mean": np.mean(temperature),
            "temperature_max": np.max(temperature),
            "temperature_std": np.std(temperature),

            "voltage_load_mean": np.mean(voltage_load),
            "current_load_mean": np.mean(current_load),

            "discharge_duration": time[-1] - time[0]
        })

features = pd.DataFrame(rows)

print("NEW FEATURE DATASET SHAPE:", features.shape)
print("\nFIRST FIVE ROWS:")
print(features.head().to_string(index=False))

print("\nMISSING VALUES:")
print(features.isnull().sum())

print("\nFEATURE SUMMARY:")
print(features.describe())

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

features.to_csv(
    output_dir / "battery_features.csv",
    index=False
)

print("\nUpdated feature dataset saved successfully!")


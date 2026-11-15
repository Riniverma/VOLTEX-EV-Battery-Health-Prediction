
from scipy.io import loadmat
import pandas as pd
from pathlib import Path

battery_files = ["B0005", "B0006", "B0007", "B0018"]
data_dir = Path("data/raw")
all_data = []

for battery_name in battery_files:
    file_path = data_dir / f"{battery_name}.mat"
    mat = loadmat(file_path)
    battery = mat[battery_name][0, 0]
    cycles = battery["cycle"][0]

    discharge_records = []

    for record_index, cycle in enumerate(cycles):
        cycle_type = cycle["type"][0]

        if cycle_type == "discharge":
            cycle_data = cycle["data"][0, 0]
            capacity = float(cycle_data["Capacity"][0, 0])

            discharge_records.append({
                "battery_id": battery_name,
                "record_index": record_index + 1,
                "capacity_Ah": capacity
            })

    df = pd.DataFrame(discharge_records)

    if df.empty:
        print(f"No discharge records found for {battery_name}")
        continue

    initial_capacity = df["capacity_Ah"].iloc[0]
    df["discharge_cycle"] = range(1, len(df) + 1)
    df["SOH_percent"] = df["capacity_Ah"] / initial_capacity * 100

    all_data.append(df)

    print(
        f"{battery_name}: {len(df)} discharge records | "
        f"Initial capacity: {initial_capacity:.4f} Ah | "
        f"Final SOH: {df['SOH_percent'].iloc[-1]:.2f}%"
    )

combined = pd.concat(all_data, ignore_index=True)
output_path = data_dir / "all_batteries_capacity_soh.csv"
combined.to_csv(output_path, index=False)

print("\nCombined dataset shape:", combined.shape)
print("Saved to:", output_path)
print(combined.groupby("battery_id").size())

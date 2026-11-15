
import pandas as pd

df = pd.read_csv("data/raw/all_batteries_capacity_soh.csv")

for battery_id, group in df.groupby("battery_id"):
    group = group.sort_values("discharge_cycle")

    print(f"\nBattery: {battery_id}")
    print("First 5 SOH values:")
    print(group[["discharge_cycle", "capacity_Ah", "SOH_percent"]].head().to_string(index=False))

    print("\nLast 5 SOH values:")
    print(group[["discharge_cycle", "capacity_Ah", "SOH_percent"]].tail().to_string(index=False))

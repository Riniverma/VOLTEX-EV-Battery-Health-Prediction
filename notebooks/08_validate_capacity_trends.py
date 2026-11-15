
import pandas as pd

df = pd.read_csv("data/raw/all_batteries_capacity_soh.csv")

df = df.sort_values(["battery_id", "discharge_cycle"])

print("BATTERY-WISE CAPACITY TREND\n")

for battery_id, group in df.groupby("battery_id"):
    group = group.sort_values("discharge_cycle")

    first_capacity = group["capacity_Ah"].iloc[0]
    last_capacity = group["capacity_Ah"].iloc[-1]

    first_soh = group["SOH_percent"].iloc[0]
    last_soh = group["SOH_percent"].iloc[-1]

    print(f"Battery: {battery_id}")
    print(f"Discharge records: {len(group)}")
    print(f"First capacity: {first_capacity:.4f} Ah")
    print(f"Last capacity:  {last_capacity:.4f} Ah")
    print(f"First SOH: {first_soh:.2f}%")
    print(f"Last SOH:  {last_soh:.2f}%")
    print(f"Capacity change: {last_capacity - first_capacity:.4f} Ah")
    print("-" * 40)

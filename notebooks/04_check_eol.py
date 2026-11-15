
import pandas as pd

df = pd.read_csv("data/raw/all_batteries_capacity_soh.csv")

EOL_THRESHOLD = 70.0

for battery_id, group in df.groupby("battery_id"):
    group = group.sort_values("discharge_cycle").reset_index(drop=True)

    below_threshold = group[group["SOH_percent"] <= EOL_THRESHOLD]

    print(f"\nBattery: {battery_id}")
    print(f"Total discharge cycles: {len(group)}")
    print(f"Initial SOH: {group['SOH_percent'].iloc[0]:.2f}%")
    print(f"Final SOH: {group['SOH_percent'].iloc[-1]:.2f}%")

    if not below_threshold.empty:
        eol_cycle = int(below_threshold["discharge_cycle"].iloc[0])
        print(f"First cycle at/below 70% SOH: {eol_cycle}")
    else:
        print("70% SOH threshold not reached in recorded data.")

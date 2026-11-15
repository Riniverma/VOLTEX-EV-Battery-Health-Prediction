
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/all_batteries_capacity_soh.csv")

# Plot capacity degradation
plt.figure(figsize=(10, 6))

for battery_id, group in df.groupby("battery_id"):
    group = group.sort_values("discharge_cycle")
    plt.plot(
        group["discharge_cycle"],
        group["capacity_Ah"],
        label=battery_id
    )

plt.title("EV Battery Capacity Degradation")
plt.xlabel("Discharge Cycle")
plt.ylabel("Capacity (Ah)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("result/all_batteries_capacity.png", dpi=300)
plt.show()

# Plot SOH
plt.figure(figsize=(10, 6))

for battery_id, group in df.groupby("battery_id"):
    group = group.sort_values("discharge_cycle")
    plt.plot(
        group["discharge_cycle"],
        group["SOH_percent"],
        label=battery_id
    )

plt.title("EV Battery State of Health")
plt.xlabel("Discharge Cycle")
plt.ylabel("SOH (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("result/all_batteries_soh.png", dpi=300)
plt.show()

print("Both plots saved in the result folder.")

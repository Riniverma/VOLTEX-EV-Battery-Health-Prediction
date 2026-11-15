
from scipy.io import loadmat
import pandas as pd

# Load NASA battery data
data = loadmat("data/raw/B0005.mat")
battery = data["B0005"][0, 0]
cycles = battery["cycle"][0]

records = []

for i, cycle in enumerate(cycles):
    cycle_type = cycle["type"][0]

    if cycle_type == "discharge":
        cycle_data = cycle["data"][0, 0]

        capacity = float(cycle_data["Capacity"][0, 0])

        records.append({
            "cycle_number": i + 1,
            "capacity_Ah": capacity
        })

df = pd.DataFrame(records)

# Calculate SOH relative to the first recorded discharge capacity
initial_capacity = df["capacity_Ah"].iloc[0]
df["SOH_percent"] = (
    df["capacity_Ah"] / initial_capacity
) * 100

print("\nFirst 10 discharge records:")
print(df.head(10).to_string(index=False))

print("\nTotal discharge records:", len(df))
print("Initial capacity (Ah):", round(initial_capacity, 4))
print("Final capacity (Ah):", round(df["capacity_Ah"].iloc[-1], 4))
print("Final SOH (%):", round(df["SOH_percent"].iloc[-1], 2))

# Save processed data
df.to_csv("data/raw/B0005_capacity_soh.csv", index=False)
print("\nSaved: data/raw/B0005_capacity_soh.csv")

import matplotlib.pyplot as plt

# Plot capacity degradation
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(df) + 1),
    df["capacity_Ah"],
    marker=".",
    markersize=3
)

plt.title("NASA B0005 Battery Capacity Degradation")
plt.xlabel("Discharge Cycle Number")
plt.ylabel("Capacity (Ah)")
plt.grid(True)
plt.tight_layout()
plt.savefig("result/B0005_capacity_degradation.png", dpi=300)
plt.show()

# Plot SOH
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(df) + 1),
    df["SOH_percent"],
    marker=".",
    markersize=3
)

plt.title("NASA B0005 Battery State of Health")
plt.xlabel("Discharge Cycle Number")
plt.ylabel("SOH (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("result/B0005_soh.png", dpi=300)
plt.show()

import matplotlib.pyplot as plt

# Capacity degradation graph
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(df) + 1),
    df["capacity_Ah"],
    marker=".",
    markersize=3
)
plt.title("B0005 Battery Capacity Degradation")
plt.xlabel("Discharge Record Number")
plt.ylabel("Capacity (Ah)")
plt.grid(True)
plt.tight_layout()
plt.savefig("result/B0005_capacity_degradation.png", dpi=300)
plt.show()

# SOH graph
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(df) + 1),
    df["SOH_percent"],
    marker=".",
    markersize=3
)
plt.title("B0005 Battery State of Health")
plt.xlabel("Discharge Record Number")
plt.ylabel("SOH (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("result/B0005_soh.png", dpi=300)
plt.show()

print("Both graphs saved in the result folder.")



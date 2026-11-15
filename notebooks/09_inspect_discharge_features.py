
from scipy.io import loadmat
from pathlib import Path

data_dir = Path("data/raw")

for battery_id in ["B0005", "B0006", "B0007", "B0018"]:
    file_path = data_dir / f"{battery_id}.mat"

    mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    battery = mat[battery_id]
    cycles = battery.cycle

    print(f"\n{'=' * 45}")
    print(f"Battery: {battery_id}")

    cycle_list = cycles if getattr(cycles, "ndim", 0) > 0 else [cycles]
    found = False

    for cycle in cycle_list:
        if getattr(cycle, "type", "") != "discharge":
            continue

        measurements = cycle.data
        print("Discharge data fields:")

        if hasattr(measurements, "_fieldnames"):
            print(measurements._fieldnames)
        else:
            print(type(measurements))

        found = True
        break

    if not found:
        print("No discharge cycle found.")

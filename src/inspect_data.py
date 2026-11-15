from scipy.io import loadmat

file_path = r"PASTE_FULL_PATH_OF_B0005.mat_HERE"

data = loadmat(file_path)

print("Dataset keys:", data.keys())

battery = data["B0005"]

print("\nBattery structure:")
print(battery.dtype)

print("\nBattery shape:")
print(battery.shape)


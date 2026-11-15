
import pandas as pd
from pathlib import Path

file_path = Path("data/raw/all_batteries_capacity_soh.csv")

df = pd.read_csv(file_path)

print("DATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nNUMERIC SUMMARY")
print(df.describe())

# Check capacity and SOH values if these columns exist
for col in ["capacity_Ah", "SOH_percent"]:
    if col in df.columns:
        print(f"\nInvalid or non-positive values in {col}:")
        print((df[col].notna() & (df[col] <= 0)).sum())

print("\nRECORDS PER BATTERY")
if "battery_id" in df.columns:
    print(df["battery_id"].value_counts())
elif "battery" in df.columns:
    print(df["battery"].value_counts())
else:
    print("Battery ID column not detected; inspect column names above.")

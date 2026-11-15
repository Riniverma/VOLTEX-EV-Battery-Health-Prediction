
import pandas as pd
from pathlib import Path

features_path = Path("data/processed/battery_features.csv")
soh_path = Path("data/raw/all_batteries_capacity_soh.csv")

features = pd.read_csv(features_path)
soh = pd.read_csv(soh_path)

# Merge using battery identity and discharge cycle
df = features.merge(
    soh[["battery_id", "discharge_cycle", "SOH_percent"]],
    on=["battery_id", "discharge_cycle"],
    how="inner",
    validate="one_to_one"
)

print("Prepared dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nRecords per battery:")
print(df["battery_id"].value_counts())
print("\nFirst five rows:")
print(df.head())

output_path = Path("data/processed/soh_model_dataset.csv")
df.to_csv(output_path, index=False)

print(f"\nSaved to {output_path}")

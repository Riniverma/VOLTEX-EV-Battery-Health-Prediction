
from pathlib import Path
import pandas as pd
import traceback

# Project folder ke andar dataset ka path
ROOT = Path(__file__).resolve().parent
DATA_FOLDER = ROOT / "data" / "raw" / "nasa_alt"

# Reports save karne ka folder
REPORT_FOLDER = ROOT / "reports"
REPORT_FOLDER.mkdir(exist_ok=True)

CHUNK_SIZE = 100_000

if not DATA_FOLDER.exists():
    raise FileNotFoundError(
        f"Dataset folder nahi mila: {DATA_FOLDER}\n"
        "Check karo ki nasa_alt folder data/raw ke andar hai."
    )

csv_files = sorted(DATA_FOLDER.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("nasa_alt folder mein koi CSV file nahi mili.")

summary = []
report_lines = []

report_lines.append("NASA ALT DATASET INSPECTION REPORT")
report_lines.append("=" * 65)
report_lines.append(f"Dataset folder: {DATA_FOLDER}")
report_lines.append(f"Total CSV files: {len(csv_files)}")
report_lines.append("")

for file in csv_files:
    print(f"Inspecting: {file.name}")

    try:
        total_rows = 0
        missing_counts = {}
        columns = []
        data_types = {}
        numeric_min = {}
        numeric_max = {}
        error = None

        for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
            if not columns:
                columns = list(chunk.columns)

            total_rows += len(chunk)

            for col in chunk.columns:
                missing_counts[col] = (
                    missing_counts.get(col, 0) + int(chunk[col].isna().sum())
                )

                if pd.api.types.is_numeric_dtype(chunk[col]):
                    values = chunk[col].dropna()

                    if not values.empty:
                        current_min = values.min()
                        current_max = values.max()

                        numeric_min[col] = min(
                            numeric_min.get(col, current_min), current_min
                        )
                        numeric_max[col] = max(
                            numeric_max.get(col, current_max), current_max
                        )

                data_types[col] = str(chunk[col].dtype)

        report_lines.append("-" * 65)
        report_lines.append(f"FILE: {file.relative_to(DATA_FOLDER)}")
        report_lines.append(f"Rows: {total_rows}")
        report_lines.append(f"Columns: {len(columns)}")
        report_lines.append(f"Column names: {', '.join(columns)}")

        report_lines.append("Missing values:")
        for col in columns:
            report_lines.append(
                f"  {col}: {missing_counts.get(col, 0)}"
            )

        report_lines.append("Data types:")
        for col in columns:
            report_lines.append(f"  {col}: {data_types.get(col, 'unknown')}")

        report_lines.append("Numeric ranges:")
        for col in numeric_min:
            report_lines.append(
                f"  {col}: {numeric_min[col]} to {numeric_max[col]}"
            )

        summary.append({
            "file": str(file.relative_to(DATA_FOLDER)),
            "rows": total_rows,
            "columns_count": len(columns),
            "columns": ", ".join(columns),
            "total_missing_values": sum(missing_counts.values()),
            "status": "Success"
        })

    except Exception as e:
        error = str(e)
        report_lines.append("-" * 65)
        report_lines.append(f"FILE: {file.name}")
        report_lines.append(f"ERROR: {error}")

        summary.append({
            "file": str(file.relative_to(DATA_FOLDER)),
            "rows": None,
            "columns_count": None,
            "columns": "",
            "total_missing_values": None,
            "status": f"Error: {error}"
        })

# Save summary CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(REPORT_FOLDER / "nasa_alt_file_summary.csv", index=False)

# Save detailed report
report_path = REPORT_FOLDER / "nasa_alt_inspection_report.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")

print("\n" + "=" * 55)
print("INSPECTION COMPLETED")
print("=" * 55)
print(f"Total CSV files: {len(csv_files)}")
print(f"Successful files: {(summary_df['status'] == 'Success').sum()}")
print(f"Reports saved in: {REPORT_FOLDER}")
print("1. nasa_alt_inspection_report.txt")
print("2. nasa_alt_file_summary.csv")

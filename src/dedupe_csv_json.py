"""
dedupe_csv_json.py

One-time cleanup — removes duplicate (Date, Time) rows from data/All_Data.csv,
keeping the FIRST occurrence of each (the more complete data block, per manual
inspection). Regenerates data/data.json from the deduped CSV afterward so both
files stay in sync (rather than deduping JSON separately, which risks drift).

Makes a timestamped backup of both files before writing anything.
Prints a dry-run summary first; only writes if DRY_RUN is False.

Run this ONCE via a manual GitHub Actions workflow_dispatch, not on a schedule.
"""

import os
import sys
import csv
import json
import shutil
from datetime import datetime
import pandas as pd

# ---------------------------------------------------------------------------
CSV_PATH = os.path.join("data", "All_Data.csv")
JSON_PATH = os.path.join("data", "data.json")
DRY_RUN = True  # set to False only after reviewing the summary below

DROP_COLUMNS_FOR_JSON = {
    "Rainfall", "Solar_Generation", "low_price", "high_price",
    "Average_Price_Rs_Per_Sqft", "QoQ_Price_Change_Percent",
    "BRPL", "BYPL", "NDPL", "NDMC", "MES"
}

# ---------------------------------------------------------------------------
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found.")
    sys.exit(1)

# All_Data.csv has no header row in the file itself (model.ipynb assigns
# names from a separate header read) — but check: if your actual file DOES
# have a header row, flip HAS_HEADER to True.
HAS_HEADER = False

if HAS_HEADER:
    df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1", low_memory=False)
else:
    # Try to infer column names the same way model.ipynb does, from a
    # sibling header source if available; otherwise use positional names.
    columns = [
        "Date", "Time", "Weekday", "Temperature", "Condition", "Humidity",
        "Wind_Speed", "Holiday", "Event", "Rainfall", "Solar_Generation",
        "low_price", "high_price", "Average_Price_Rs_Per_Sqft",
        "QoQ_Price_Change_Percent", "Load", "BRPL", "BYPL", "NDPL", "NDMC", "MES"
    ]
    df = pd.read_csv(CSV_PATH, header=None, names=columns, encoding="ISO-8859-1", low_memory=False)

total_before = len(df)
dup_mask = df.duplicated(subset=["Date", "Time"], keep="first")
dup_count = dup_mask.sum()

print(f"--- Duplicate check on {CSV_PATH} ---")
print(f"Total rows before: {total_before}")
print(f"Duplicate (Date, Time) rows found (to be removed): {dup_count}")

if dup_count > 0:
    print("\nSample of duplicate rows that would be removed (up to 5):")
    print(df[dup_mask][["Date", "Time", "Temperature", "Condition", "Load"]].head(5).to_string(index=False))

df_deduped = df[~dup_mask].reset_index(drop=True)
total_after = len(df_deduped)

print(f"\nTotal rows after dedup would be: {total_after}")

if dup_count == 0:
    print("\nNo duplicates found. Nothing to do.")
    sys.exit(0)

if DRY_RUN:
    print(f"\nDRY_RUN is True — no files were modified.")
    print(f"Review the numbers above. If correct, set DRY_RUN = False and re-run.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Actually write: backup first, then overwrite CSV, then regenerate JSON
# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_csv = f"{CSV_PATH}.backup_{timestamp}"
backup_json = f"{JSON_PATH}.backup_{timestamp}" if os.path.exists(JSON_PATH) else None

shutil.copy2(CSV_PATH, backup_csv)
print(f"Backed up original CSV to: {backup_csv}")

if backup_json:
    shutil.copy2(JSON_PATH, backup_json)
    print(f"Backed up original JSON to: {backup_json}")

# Write deduped CSV back out in the SAME format as before (no header row,
# matching how the daily script appends to it)
df_deduped.to_csv(CSV_PATH, index=False, header=False, encoding="ISO-8859-1")
print(f"Wrote deduped CSV: {total_after} rows (removed {dup_count}).")

# Regenerate data.json from the clean CSV, same conversion logic as the
# daily script uses, so both files are guaranteed in sync
def convert_types(row):
    return {
        "Date": row["Date"],
        "Time": row["Time"],
        "Weekday": row["Weekday"],
        "Temperature": float(row["Temperature"]) if pd.notna(row["Temperature"]) else None,
        "Condition": row["Condition"],
        "Humidity": int(row["Humidity"]) if pd.notna(row["Humidity"]) else None,
        "Wind_Speed": float(row["Wind_Speed"]) if pd.notna(row["Wind_Speed"]) else None,
        "Holiday": bool(row["Holiday"]) if pd.notna(row["Holiday"]) else False,
        "Event": row["Event"] if pd.notna(row["Event"]) and row["Event"] not in ["No", ""] else None,
        "Load": float(row["Load"]) if pd.notna(row["Load"]) else None,
    }

json_rows = [
    convert_types(row) for _, row in df_deduped.drop(columns=list(DROP_COLUMNS_FOR_JSON), errors="ignore").iterrows()
]

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(json_rows, f, indent=4)

print(f"Regenerated {JSON_PATH}: {len(json_rows)} rows.")
print("\nDedup complete. Backups kept in data/ folder in case you need to revert.")

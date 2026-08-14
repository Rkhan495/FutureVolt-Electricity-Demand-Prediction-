"""
strip_stray_headers.py

One-time cleanup — removes any row(s) in data/All_Data.csv where the "Date"
column literally contains the string "Date" (i.e. an accidentally embedded
header row, likely from a manual Excel/Sheets edit somewhere in the file's
history).

Makes a timestamped backup before writing. Dry-run first; only writes if
DRY_RUN is set to False.

Run this ONCE via a manual GitHub Actions workflow_dispatch, not on a schedule.
"""

import os
import sys
import shutil
from datetime import datetime
import pandas as pd

CSV_PATH = os.path.join("data", "All_Data.csv")
DRY_RUN = False  # set to False only after reviewing the summary below

COLUMNS = [
    "Date", "Time", "Weekday", "Temperature", "Condition", "Humidity",
    "Wind_Speed", "Holiday", "Event", "Rainfall", "Solar_Generation",
    "low_price", "high_price", "Average_Price_Rs_Per_Sqft",
    "QoQ_Price_Change_Percent", "Load", "BRPL", "BYPL", "NDPL", "NDMC", "MES"
]

if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH, header=None, names=COLUMNS, encoding="ISO-8859-1", low_memory=False)
total_before = len(df)

stray_mask = df["Date"] == "Date"
stray_count = stray_mask.sum()

print(f"--- Stray header row check on {CSV_PATH} ---")
print(f"Total rows: {total_before}")
print(f"Stray embedded header rows found: {stray_count}")

if stray_count > 0:
    print("\nRow number(s) containing stray headers (0-indexed):")
    print(df[stray_mask].index.tolist())

if stray_count == 0:
    print("\nNo stray header rows found. Nothing to do.")
    sys.exit(0)

df_clean = df[~stray_mask].reset_index(drop=True)
total_after = len(df_clean)

print(f"\nTotal rows after cleanup would be: {total_after}")

if DRY_RUN:
    print(f"\nDRY_RUN is True — no files were modified.")
    print(f"Review the numbers above. If correct, set DRY_RUN = False and re-run.")
    sys.exit(0)

# ---------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"{CSV_PATH}.backup_strayheader_{timestamp}"
shutil.copy2(CSV_PATH, backup_path)
print(f"Backed up original CSV to: {backup_path}")

df_clean.to_csv(CSV_PATH, index=False, header=False, encoding="ISO-8859-1")
print(f"Wrote cleaned CSV: {total_after} rows (removed {stray_count} stray header row(s)).")
print("\nCleanup complete. Backup kept in data/ folder in case you need to revert.")

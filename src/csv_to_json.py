"""
csv_to_json.py

Regenerates data/data.json directly from data/All_Data.csv, entirely inside
the repo via GitHub Actions — no manual download/edit/upload needed, and no
25MB upload-size problem since this runs server-side and commits the result
via git, not through the GitHub web upload UI.

Safe to run anytime; it always rebuilds JSON fresh from the current CSV,
so the two files can never drift out of sync as long as this is the only
way data.json gets updated.
"""

import os
import sys
import json
import pandas as pd

CSV_PATH = os.path.join("data", "All_Data.csv")
JSON_PATH = os.path.join("data", "data.json")

DROP_COLUMNS_FOR_JSON = {
    "Rainfall", "Solar_Generation", "low_price", "high_price",
    "Average_Price_Rs_Per_Sqft", "QoQ_Price_Change_Percent",
    "BRPL", "BYPL", "NDPL", "NDMC", "MES"
}

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
print(f"Loaded {len(df)} rows from {CSV_PATH}")

# Defensive: strip any stray embedded header row(s) that may have gotten
# saved into the data itself (e.g. from a manual Excel/Sheets edit)
before = len(df)
df = df[df["Date"] != "Date"].reset_index(drop=True)
removed = before - len(df)
if removed:
    print(f"Removed {removed} stray embedded header row(s) found in the data.")


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
    convert_types(row)
    for _, row in df.drop(columns=list(DROP_COLUMNS_FOR_JSON), errors="ignore").iterrows()
]

with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(json_rows, f, indent=4)

print(f"Wrote {len(json_rows)} rows to {JSON_PATH}")

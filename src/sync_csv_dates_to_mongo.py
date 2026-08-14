"""
sync_csv_dates_to_mongo.py

Reads specific date(s) from data/All_Data.csv (your CSV backup) and
upserts them into the MongoDB 'data' collection (the one your Node.js
backend's LoadData model queries for the Historical Data page).

Useful for exactly this situation: CSV/JSON have a date's data, but it
never made it into MongoDB (e.g. because that day's scrape ran before a
fix was deployed, or before "is_current" logic caught it).

Per-date REPLACE semantics — deletes only that date's existing docs (if
any) before inserting fresh ones, so it's safe to re-run and won't create
duplicates.

Dry-run first; only writes if DRY_RUN is False.
"""

import os
import sys
import pandas as pd
import pymongo
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — list the date(s), in DD-MM-YYYY format (matching your CSV), that
# need to be synced from CSV into MongoDB's 'data' collection.
# ---------------------------------------------------------------------------
TARGET_DATES = ["12-08-2026"]
CSV_PATH = os.path.join("data", "All_Data.csv")
COLLECTION_NAME = "data"
DRY_RUN = True

COLUMNS = [
    "Date", "Time", "Weekday", "Temperature", "Condition", "Humidity",
    "Wind_Speed", "Holiday", "Event", "Rainfall", "Solar_Generation",
    "low_price", "high_price", "Average_Price_Rs_Per_Sqft",
    "QoQ_Price_Change_Percent", "Load", "BRPL", "BYPL", "NDPL", "NDMC", "MES"
]

# ---------------------------------------------------------------------------
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    print("ERROR: MONGODB_URI not found in environment variables")
    sys.exit(1)

client = pymongo.MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
client.admin.command("ping")
print("Successfully connected to MongoDB!")

db = client.FutureVolt
collection = db[COLLECTION_NAME]

if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH, header=None, names=COLUMNS, encoding="ISO-8859-1", low_memory=False)
df = df[df["Date"] != "Date"].reset_index(drop=True)  # strip stray header rows if present

for target_date in TARGET_DATES:
    csv_rows = df[df["Date"] == target_date]
    mongo_count = collection.count_documents({"Date": target_date})

    print(f"\n--- {target_date} ---")
    print(f"Rows in CSV: {len(csv_rows)}")
    print(f"Documents currently in MongoDB '{COLLECTION_NAME}': {mongo_count}")

    if csv_rows.empty:
        print(f"No CSV data for {target_date} — nothing to sync.")
        continue

    documents = []
    for _, row in csv_rows.iterrows():
        documents.append({
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
        })

    print(f"Would replace MongoDB's {mongo_count} existing doc(s) with {len(documents)} fresh doc(s) from CSV.")

    if DRY_RUN:
        print("DRY_RUN is True — no changes made.")
        continue

    collection.delete_many({"Date": target_date})
    collection.insert_many(documents)
    print(f"Synced {len(documents)} documents for {target_date} into '{COLLECTION_NAME}'.")

if DRY_RUN:
    print("\nAll dates were dry-run only. Review above, set DRY_RUN = False, and re-run to actually sync.")
else:
    print("\nSync complete.")

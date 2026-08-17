"""
sync_mongo_dates_to_csv.py

Reverse of sync_csv_dates_to_mongo.py — reads specific date(s) from the
MongoDB 'data' collection and appends them into data/All_Data.csv, for
cases where Mongo has data that the CSV backup is missing.

Note: MongoDB documents only store Date, Time, Weekday, Temperature,
Condition, Humidity, Wind_Speed, Holiday, Event, Load — NOT Rainfall,
Solar_Generation, or real-estate fields. This script reconstructs those
from your existing forecast CSVs (deterministic, date-based lookups):
monthly forecast totals divided by days-in-month, same pattern used for
Solar_Generation and Rainfall elsewhere in the pipeline.

Skips (Date, Time) pairs already present in the CSV — safe to re-run.
Dry-run first; only writes if DRY_RUN is False.
"""

import os
import sys
import calendar
import subprocess
import pandas as pd
import pymongo
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
TARGET_DATES = ["13-08-2026"]
CSV_PATH = os.path.join("data", "All_Data.csv")
RAINFALL_PATH = "rainfall_data_forecast.csv"
SOLAR_PATH = "solar_data_forecast.csv"
REAL_ESTATE_PATH = "real_estate_price_forecast.csv"
COLLECTION_NAME = "data"
DRY_RUN = False  # confirm all three lookups succeed before flipping to False

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

existing_df = pd.read_csv(CSV_PATH, header=None, names=COLUMNS, encoding="ISO-8859-1", low_memory=False)
existing_df = existing_df[existing_df["Date"] != "Date"].reset_index(drop=True)
existing_pairs = set(zip(existing_df["Date"], existing_df["Time"]))

for path, label in [(RAINFALL_PATH, "rainfall"), (SOLAR_PATH, "solar"), (REAL_ESTATE_PATH, "real estate")]:
    if not os.path.exists(path):
        print(f"ERROR: {label} file not found: {os.path.abspath(path)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in CWD: {os.listdir('.')}")
        sys.exit(1)

rainfall_data = pd.read_csv(RAINFALL_PATH)
rainfall_data["Date"] = pd.to_datetime(rainfall_data["Date"], format="%d-%m-%Y")  # confirmed DD-MM-YYYY
solar_data = pd.read_csv(SOLAR_PATH)
solar_data["Date"] = pd.to_datetime(solar_data["Date"], format="%Y-%m-%d")  # confirmed YYYY-MM-DD
real_estate_data = pd.read_csv(REAL_ESTATE_PATH)
real_estate_data["date"] = pd.to_datetime(real_estate_data["date"], format="%d-%m-%Y")  # confirmed DD-MM-YYYY

print(f"\n[DEBUG] Loaded rainfall_data: {len(rainfall_data)} rows from {os.path.abspath(RAINFALL_PATH)}")
print(f"[DEBUG] rainfall_data columns: {list(rainfall_data.columns)}")
print(f"[DEBUG] rainfall_data 'Date' sample (parsed): {rainfall_data['Date'].head(5).tolist()}")

print(f"\n[DEBUG] Loaded solar_data: {len(solar_data)} rows from {os.path.abspath(SOLAR_PATH)}")
print(f"[DEBUG] solar_data columns: {list(solar_data.columns)}")
print(f"[DEBUG] solar_data 'Date' FULL list: {solar_data['Date'].tolist()}")
try:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    print(f"[DEBUG] Current git branch: {branch}")
    print(f"[DEBUG] Current git commit: {commit}")
except Exception as e:
    print(f"[DEBUG] Could not determine git branch/commit: {e}")

print(f"\n[DEBUG] Loaded real_estate_data: {len(real_estate_data)} rows from {os.path.abspath(REAL_ESTATE_PATH)}")
print(f"[DEBUG] real_estate_data columns: {list(real_estate_data.columns)}")
print(f"[DEBUG] real_estate_data 'date' sample values: {real_estate_data['date'].head(5).tolist()}")


def get_rainfall(year, month):
    """Monthly forecast total (mm) divided by days-in-month -> per-day estimate."""
    print(f"[DEBUG] Looking up rainfall_data where year == {year} and month == {month}")
    monthly = rainfall_data[(rainfall_data["Date"].dt.year == year) & (rainfall_data["Date"].dt.month == month)]
    print(f"[DEBUG] Matches found: {len(monthly)}")
    if monthly.empty:
        return None
    last_day = calendar.monthrange(year, month)[1]
    return round(monthly["Rainfall"].values[0], 2) / last_day


def get_solar_generation(year, month):
    print(f"[DEBUG] Looking up solar_data where year == {year} and month == {month}")
    monthly = solar_data[(solar_data["Date"].dt.year == year) & (solar_data["Date"].dt.month == month)]
    print(f"[DEBUG] Matches found: {len(monthly)}")
    if monthly.empty:
        return None
    last_day = calendar.monthrange(year, month)[1]
    return round(monthly["Forecasted Solar Generation"].values[0], 2) / last_day


def get_real_estate(year, date_ts):
    print(f"[DEBUG] Looking up real_estate_data where year == {year} and quarter == {date_ts.quarter}")
    mask = (real_estate_data["date"].dt.year == year) & (real_estate_data["date"].dt.quarter == date_ts.quarter)
    q = real_estate_data[mask]
    print(f"[DEBUG] Matches found: {len(q)}")
    if q.empty:
        return None
    return (
        q["low_price_pred"].values.item(),
        q["high_price_pred"].values.item(),
        q["Average_Price"].values.item(),
        q["QoQ_Price_Change_Percent"].values.item(),
    )


all_new_rows = []

for target_date in TARGET_DATES:
    day, month, year = map(int, target_date.split("-"))
    mongo_docs = list(collection.find({"Date": target_date}).sort("Time", 1))

    print(f"\n--- {target_date} ---")
    print(f"Documents in MongoDB '{COLLECTION_NAME}': {len(mongo_docs)}")

    already_in_csv = sum(1 for doc in mongo_docs if (doc["Date"], doc["Time"]) in existing_pairs)
    print(f"Already present in CSV: {already_in_csv}")
    print(f"Would be added to CSV: {len(mongo_docs) - already_in_csv}")

    if not mongo_docs:
        print(f"No MongoDB data for {target_date} — nothing to sync.")
        continue

    date_ts = pd.to_datetime(f"{year}-{month}-{day}")
    rainfall = get_rainfall(year, month)
    solar_generation = get_solar_generation(year, month)
    real_estate = get_real_estate(year, date_ts)

    if rainfall is None:
        print(f"WARNING: no rainfall forecast data found for {target_date} — skipping this date.")
        continue
    if solar_generation is None:
        print(f"WARNING: no solar generation data found for {target_date} — skipping this date.")
        continue
    if real_estate is None:
        print(f"WARNING: no real-estate forecast data found for {target_date} — skipping this date.")
        continue

    low_price, high_price, avg_price, qoq_price = real_estate

    for doc in mongo_docs:
        if (doc["Date"], doc["Time"]) in existing_pairs:
            continue
        all_new_rows.append({
            "Date": doc["Date"], "Time": doc["Time"], "Weekday": doc["Weekday"],
            "Temperature": doc["Temperature"], "Condition": doc["Condition"],
            "Humidity": doc["Humidity"], "Wind_Speed": doc["Wind_Speed"],
            "Holiday": int(doc["Holiday"]), "Event": doc["Event"] or "No",
            "Rainfall": round(rainfall, 2),
            "Solar_Generation": round(solar_generation, 2),
            "low_price": round(low_price, 2), "high_price": round(high_price, 2),
            "Average_Price_Rs_Per_Sqft": round(avg_price, 2),
            "QoQ_Price_Change_Percent": round(qoq_price, 2),
            "Load": doc["Load"],
            "BRPL": None, "BYPL": None, "NDPL": None, "NDMC": None, "MES": None,
        })

print(f"\nTotal new rows to append across all target dates: {len(all_new_rows)}")

if not all_new_rows:
    print("Nothing to add. CSV already has everything MongoDB has for these dates (or a lookup failed above).")
    sys.exit(0)

if DRY_RUN:
    print("\nDRY_RUN is True — no changes made.")
    print("Sample of rows that would be added (up to 3):")
    for r in all_new_rows[:3]:
        print(f"  {r['Date']} {r['Time']} Load={r['Load']} Rainfall={r['Rainfall']}")
    print("\nReview above. If correct, set DRY_RUN = False and re-run.")
    sys.exit(0)

new_df = pd.DataFrame(all_new_rows, columns=COLUMNS)
new_df.to_csv(CSV_PATH, index=False, mode="a", header=False, encoding="ISO-8859-1")
print(f"\nAppended {len(new_df)} rows to {CSV_PATH}.")
print("Run csv_to_json.py (or the 'Regenerate JSON from CSV' workflow) next to sync data.json.")

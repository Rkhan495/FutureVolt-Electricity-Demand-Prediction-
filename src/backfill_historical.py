"""
backfill_historical.py

One-time backfill script — pulls REAL historical weather for New Delhi from
Open-Meteo's free historical API, merges it with existing solar / real estate /
holiday data, runs it through the existing trained model, and inserts the
results into MongoDB (db.data) for the date range specified below.

This is completely separate from Electricity_Demand_Prediction.py (the daily
scraper) — it does not touch or modify that script or its schedule.

Run this ONCE, locally or via a manual GitHub Actions dispatch, not on a schedule.
"""

import os
import sys
import gzip
import pickle
import calendar
import requests
import numpy as np
import pandas as pd
import pymongo
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — edit these two dates if you need a different backfill range
# ---------------------------------------------------------------------------
START_DATE = "2026-04-20"
END_DATE = "2026-08-10"
LATITUDE = 28.6139   # New Delhi
LONGITUDE = 77.2090

# ---------------------------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------------------------
mongodb_uri = os.getenv("MONGODB_URI")
if not mongodb_uri:
    print("ERROR: MONGODB_URI not found in environment variables")
    sys.exit(1)

client = pymongo.MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
client.admin.command("ping")
print("Successfully connected to MongoDB!")

db = client.FutureVolt
target_collection = db["data"]  # historical collection — NOT FutureData

# ---------------------------------------------------------------------------
# Step 1: Pull real historical hourly weather from Open-Meteo (free, no key)
# ---------------------------------------------------------------------------
print(f"Fetching historical weather from Open-Meteo: {START_DATE} to {END_DATE}...")

om_url = "https://archive-api.open-meteo.com/v1/archive"
om_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
    "timezone": "Asia/Kolkata",
}

resp = requests.get(om_url, params=om_params, timeout=60)
resp.raise_for_status()
om_data = resp.json()

hourly = om_data["hourly"]
weather_df = pd.DataFrame({
    "datetime": pd.to_datetime(hourly["time"]),
    "Temperature": hourly["temperature_2m"],
    "Humidity": hourly["relative_humidity_2m"],
    "Rainfall": hourly["precipitation"],
    "Wind_Speed": hourly["wind_speed_10m"],
    "weather_code": hourly["weather_code"],
})

print(f"Fetched {len(weather_df)} hourly records from Open-Meteo.")

# ---------------------------------------------------------------------------
# Step 2: Map Open-Meteo WMO weather codes -> approximate Condition strings
# (matched loosely to the vocabulary your scraper produced from timeanddate.com)
# ---------------------------------------------------------------------------
WMO_CONDITION_MAP = {
    0: "Sunny", 1: "Mostly Sunny", 2: "Partly Cloudy", 3: "Cloudy",
    45: "Fog", 48: "Fog",
    51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
    56: "Light Drizzle", 57: "Heavy Drizzle",
    61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
    66: "Light Rain", 67: "Heavy Rain",
    71: "Light Snow", 73: "Snow", 75: "Heavy Snow", 77: "Snow",
    80: "Light Rain", 81: "Rain", 82: "Heavy Rain",
    85: "Light Snow", 86: "Heavy Snow",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}

weather_df["Condition"] = weather_df["weather_code"].map(WMO_CONDITION_MAP).fillna("Cloudy")

# ---------------------------------------------------------------------------
# Step 3: Load supporting datasets (same files the daily script uses)
# ---------------------------------------------------------------------------
holiday_data_path = os.path.join("data", "Holidays.csv")
holiday_data = pd.read_csv(holiday_data_path)

solar_data_path = os.path.join("solar_data_forecast.csv")
solar_data = pd.read_csv(solar_data_path)

real_estate_data_path = os.path.join("real_estate_price_forecast.csv")
real_estate_data = pd.read_csv(real_estate_data_path)
real_estate_data["date"] = pd.to_datetime(real_estate_data["date"], dayfirst=True)

with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)


def unique_event_concat(events):
    words = "/".join(events).split("/")
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    return "/".join(unique_words)


holiday_data_grouped = holiday_data.groupby(
    ["Day", "Month", "Year"], as_index=False
).agg({"Holiday": "first", "Event": unique_event_concat})


def cyclic_encoding(value, max_value):
    sin_value = np.sin(2 * np.pi * value / max_value)
    cos_value = np.cos(2 * np.pi * value / max_value)
    return sin_value, cos_value


# ---------------------------------------------------------------------------
# Step 4: Build feature rows, predict Load, and insert into MongoDB
# ---------------------------------------------------------------------------
documents = []
skipped = 0

for _, row in weather_df.iterrows():
    dt = row["datetime"]
    day, month, year = dt.day, dt.month, dt.year
    hour = dt.hour
    weekday = dt.weekday()
    day_of_year = dt.timetuple().tm_yday

    # Holiday / Event lookup
    matched_row = holiday_data_grouped[
        (holiday_data_grouped["Day"] == day)
        & (holiday_data_grouped["Month"] == month)
        & (holiday_data_grouped["Year"] == year)
    ]
    if not matched_row.empty:
        holiday = matched_row["Holiday"].values[0]
        event = matched_row["Event"].values[0]
    else:
        holiday = 0
        event = "No"
        if weekday in [5, 6]:
            holiday = 1
            event = "Weekend"

    # Solar generation lookup (monthly total / days in month)
    month_start = f"{year}-{month:02d}-01"
    monthly_solar_data = solar_data[solar_data["Date"] == month_start]
    if monthly_solar_data.empty:
        skipped += 1
        continue
    last_day = calendar.monthrange(year, month)[1]
    solar_generation = round(monthly_solar_data["Forecasted Solar Generation"].values[0], 2) / last_day

    # Real estate lookup (quarterly)
    date_ts = pd.to_datetime(f"{year}-{month}-{day}")
    quarter_mask = (real_estate_data["date"].dt.year == year) & (
        real_estate_data["date"].dt.quarter == date_ts.quarter
    )
    quarter_real_estate_data = real_estate_data[quarter_mask]
    if quarter_real_estate_data.empty:
        skipped += 1
        continue
    low_price = quarter_real_estate_data["low_price_pred"].values.item()
    high_price = quarter_real_estate_data["high_price_pred"].values.item()
    avg_price = quarter_real_estate_data["Average_Price"].values.item()
    qoq_price = quarter_real_estate_data["QoQ_Price_Change_Percent"].values.item()

    # Cyclic encodings
    hour_sin, hour_cos = cyclic_encoding(hour, 24)
    weekday_sin, weekday_cos = cyclic_encoding(weekday, 7)
    month_sin, month_cos = cyclic_encoding(month, 12)
    dayofyear_sin, dayofyear_cos = cyclic_encoding(day_of_year, 365)

    temp = round(float(row["Temperature"]), 2)
    humidity = int(round(row["Humidity"]))
    wind_speed = round(float(row["Wind_Speed"]), 2)
    rain = round(float(row["Rainfall"]), 2)
    condition = row["Condition"]
    temp_x_hour = temp * hour

    features = pd.DataFrame({
        "Weekday": [weekday], "Temperature": [temp], "Condition": [condition],
        "Humidity": [humidity], "Wind_Speed": [wind_speed], "Holiday": [holiday],
        "Event": [event], "Rainfall": [rain], "Solar_Generation": [round(solar_generation, 2)],
        "low_price": [round(low_price, 2)], "high_price": [round(high_price, 2)],
        "Average_Price_Rs_Per_Sqft": [round(avg_price, 2)],
        "QoQ_Price_Change_Percent": [round(qoq_price, 2)], "Day": [day], "Month": [month],
        "Year": [year], "DayOfYear": [day_of_year], "Hour": [hour], "Hour_sin": [hour_sin],
        "Hour_cos": [hour_cos], "Weekday_sin": [weekday_sin], "Weekday_cos": [weekday_cos],
        "Month_sin": [month_sin], "Month_cos": [month_cos], "DayOfYear_sin": [dayofyear_sin],
        "DayOfYear_cos": [dayofyear_cos], "temp_x_hour": [temp_x_hour],
    })

    prediction = model.predict(features)
    load = float(np.round(prediction, 3)[0])

    hour_next = (hour + 1) % 24

    document = {
        "Date": f"{day:02d}-{month:02d}-{year}",
        "Time": f"{hour:02d}-00:{hour_next:02d}:00",
        "Weekday": calendar.day_name[weekday],
        "Temperature": temp,
        "Condition": condition,
        "Humidity": humidity,
        "Wind_Speed": wind_speed,
        "Holiday": bool(holiday),
        "Event": event if event not in ["No", ""] else None,
        "Load": load,
        "Source": "backfill_openmeteo",  # tag so you can distinguish from live-scraped rows
    }
    documents.append(document)

print(f"Built {len(documents)} documents ({skipped} rows skipped due to missing solar/real-estate coverage).")

# ---------------------------------------------------------------------------
# Step 5: Insert into MongoDB (skips duplicates on Date+Time if already present)
# ---------------------------------------------------------------------------
if documents:
    inserted = 0
    for doc in documents:
        exists = target_collection.find_one({"Date": doc["Date"], "Time": doc["Time"]})
        if not exists:
            target_collection.insert_one(doc)
            inserted += 1
    print(f"Inserted {inserted} new documents into db.data (skipped {len(documents) - inserted} already-existing rows).")
else:
    print("No documents to insert.")

print("Backfill complete.")

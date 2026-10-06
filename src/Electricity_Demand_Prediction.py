"""
Electricity_Demand_Prediction.py (v3 — Open-Meteo based, no browser/Selenium)

Replaces the timeanddate.com Selenium scraper entirely with Open-Meteo's
free forecast API. This removes the Cloudflare bot-detection arms race,
Chrome/chromedriver version mismatches, and "table not found" failures
that repeatedly broke the scraper-based version.

Fetches ~8 days of hourly forecast (today through +7 days), builds the
same feature set used in training, predicts Load via the existing model,
and upserts into MongoDB (per-hour, never wiping a whole date) plus
writes "tomorrow" into All_Data.csv the same way the old script did.
"""

import sys
import calendar
import requests
import pandas as pd
import numpy as np
import gzip
import pickle
import json
import os
import pymongo
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

LATITUDE = 28.6139
LONGITUDE = 77.2090
FORECAST_DAYS = 8  # today + next 7 days

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

# ---------------------------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------------------------
try:
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")
    print(f"Connecting to MongoDB at: {mongodb_uri[:20]}...")
    client = pymongo.MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except pymongo.errors.ConnectionFailure as e:
    print(f"MongoDB connection failed: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)

db = client.FutureVolt
collection = db["FutureData"]


def upsert_date_documents(documents):
    """Per-(Date, Time) upsert — never deletes a whole date, so partial
    updates never wipe previously-good hours for that date."""
    for doc in documents:
        collection.update_one(
            {"Date": doc["Date"], "Time": doc["Time"]},
            {"$set": doc},
            upsert=True,
        )


def create_document(data_row):
    return {
        "Date": data_row["Date"],
        "Time": data_row["Time"],
        "Weekday": data_row["Weekday"],
        "Temperature": float(data_row["Temperature"]),
        "Condition": data_row["Condition"],
        "Humidity": int(data_row["Humidity"]),
        "Wind_Speed": float(data_row["Wind_Speed"]),
        "Holiday": bool(int(data_row["Holiday"])),
        "Event": data_row["Event"] if data_row["Event"] not in ['No', ''] else None,
        "Load": float(data_row["Load"]),
    }


# ---------------------------------------------------------------------------
# Load supporting data + model
# ---------------------------------------------------------------------------
holiday_data = pd.read_csv(os.path.join("data", "Holidays.csv"))
solar_data = pd.read_csv("solar_data_forecast.csv")
solar_data['Date'] = pd.to_datetime(solar_data['Date'], format="%Y-%m-%d")
real_estate_data = pd.read_csv("real_estate_price_forecast.csv")
real_estate_data['date'] = pd.to_datetime(real_estate_data['date'], format="%d-%m-%Y")

with gzip.open('model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

csv_file = os.path.join("data", "All_Data.csv")
json_file = os.path.join("data", "data.json")

CSV_COLUMNS = [
    'Date', 'Time', "Weekday", "Temperature", "Condition", "Humidity", "Wind_Speed", "Holiday", "Event",
    "Rainfall", "Solar_Generation", "low_price", "high_price", "Average_Price_Rs_Per_Sqft",
    "QoQ_Price_Change_Percent", 'Load', 'BRPL', 'BYPL', 'NDPL', 'NDMC', 'MES'
]


def unique_event_concat(events):
    words = "/".join(events).split("/")
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    return "/".join(unique_words)


holiday_data_grouped = holiday_data.groupby(['Day', 'Month', 'Year'], as_index=False).agg({
    'Holiday': 'first',
    'Event': unique_event_concat
})


def cyclic_encoding(value, max_value):
    return np.sin(2 * np.pi * value / max_value), np.cos(2 * np.pi * value / max_value)


def get_holiday_event(day, month, year, weekday):
    matched_row = holiday_data_grouped[
        (holiday_data_grouped['Day'] == day) &
        (holiday_data_grouped['Month'] == month) &
        (holiday_data_grouped['Year'] == year)
    ]
    if not matched_row.empty:
        return matched_row['Holiday'].values[0], matched_row['Event'].values[0]
    holiday = 0
    event = 'No'
    if weekday in [5, 6]:
        holiday = 1
        event = 'Weekend'
    return holiday, event


def get_solar_generation(year, month):
    monthly = solar_data[(solar_data['Date'].dt.year == year) & (solar_data['Date'].dt.month == month)]
    if monthly.empty:
        return None
    last_day = calendar.monthrange(year, month)[1]
    return round(monthly['Forecasted Solar Generation'].values[0], 2) / last_day


def get_real_estate(year, date_ts):
    mask = (real_estate_data['date'].dt.year == year) & (real_estate_data['date'].dt.quarter == date_ts.quarter)
    q = real_estate_data[mask]
    if q.empty:
        return None
    return (
        q['low_price_pred'].values.item(),
        q['high_price_pred'].values.item(),
        q['Average_Price'].values.item(),
        q['QoQ_Price_Change_Percent'].values.item(),
    )


def predict_load(weekday, temp, condition, humidity, wind_speed, holiday, event, rain,
                  solar_generation, low_price, high_price, avg_price, qoq_price,
                  day, month, year, day_of_year, hour):
    hour_sin, hour_cos = cyclic_encoding(hour, 24)
    weekday_sin, weekday_cos = cyclic_encoding(weekday, 7)
    month_sin, month_cos = cyclic_encoding(month, 12)
    dayofyear_sin, dayofyear_cos = cyclic_encoding(day_of_year, 365)
    temp_x_hour = round(temp, 2) * hour

    features = pd.DataFrame({
        "Weekday": [weekday], "Temperature": [round(temp, 2)], "Condition": [condition],
        "Humidity": [humidity], "Wind_Speed": [wind_speed], "Holiday": [holiday],
        "Event": [event], "Rainfall": [rain], "Solar_Generation": [round(solar_generation, 2)],
        "low_price": [round(low_price, 2)], "high_price": [round(high_price, 2)],
        "Average_Price_Rs_Per_Sqft": [round(avg_price, 2)], "QoQ_Price_Change_Percent": [round(qoq_price, 2)],
        "Day": [day], "Month": [month], "Year": [year], "DayOfYear": [day_of_year], "Hour": [hour],
        "Hour_sin": [hour_sin], "Hour_cos": [hour_cos], "Weekday_sin": [weekday_sin], "Weekday_cos": [weekday_cos],
        "Month_sin": [month_sin], "Month_cos": [month_cos], "DayOfYear_sin": [dayofyear_sin],
        "DayOfYear_cos": [dayofyear_cos], "temp_x_hour": [temp_x_hour],
    })
    prediction = model.predict(features)
    return np.round(prediction, 3)[0]


# ---------------------------------------------------------------------------
# Fetch forecast from Open-Meteo (replaces Selenium scraping entirely)
# ---------------------------------------------------------------------------
print(f"Fetching {FORECAST_DAYS}-day hourly forecast from Open-Meteo...")

om_url = "https://api.open-meteo.com/v1/forecast"
om_params = {
    "latitude": LATITUDE, "longitude": LONGITUDE,
    "forecast_days": FORECAST_DAYS,
    "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
    "timezone": "Asia/Kolkata",
}

last_error = None
om_data = None
for attempt in range(3):
    try:
        resp = requests.get(om_url, params=om_params, timeout=60)
        resp.raise_for_status()
        om_data = resp.json()
        break
    except Exception as e:
        last_error = e
        print(f"Open-Meteo attempt {attempt + 1} failed: {e}")

if om_data is None:
    print(f"ERROR: Could not fetch forecast from Open-Meteo after 3 attempts: {last_error}")
    sys.exit(1)

hourly = om_data["hourly"]
weather_df = pd.DataFrame({
    "datetime": pd.to_datetime(hourly["time"]),
    "Temperature": hourly["temperature_2m"],
    "Humidity": hourly["relative_humidity_2m"],
    "Rainfall": hourly["precipitation"],
    "Wind_Speed": hourly["wind_speed_10m"],
    "weather_code": hourly["weather_code"],
})
weather_df["Condition"] = weather_df["weather_code"].map(WMO_CONDITION_MAP).fillna("Cloudy")
print(f"Fetched {len(weather_df)} hourly records covering {weather_df['datetime'].dt.date.nunique()} days.")

# ---------------------------------------------------------------------------
# Build documents per date, upsert into Mongo, write "tomorrow" to CSV
# ---------------------------------------------------------------------------
today = datetime.now()
tomorrow_date_obj = (today + timedelta(days=1)).date()

by_date = {}
for _, row in weather_df.iterrows():
    dt = row["datetime"]
    date_str = dt.strftime("%d-%m-%Y")
    by_date.setdefault(date_str, []).append(row)

successfully_processed_dates = set()
tomorrow_csv_rows = []

for date_str, rows in by_date.items():
    day, month, year = map(int, date_str.split("-"))
    full_date = datetime(year, month, day)
    weekday = full_date.weekday()
    day_of_year = full_date.timetuple().tm_yday
    date_ts = pd.to_datetime(f"{year}-{month}-{day}")

    solar_generation = get_solar_generation(year, month)
    real_estate = get_real_estate(year, date_ts)
    if solar_generation is None or real_estate is None:
        print(f"Skipping {date_str}: missing solar/real-estate reference data.")
        continue
    low_price, high_price, avg_price, qoq_price = real_estate

    date_documents = []
    date_csv_rows = []

    for row in rows:
        dt = row["datetime"]
        hour = dt.hour
        holiday, event = get_holiday_event(day, month, year, weekday)

        temp = round(float(row["Temperature"]), 2)
        humidity = int(round(row["Humidity"]))
        wind_speed = round(float(row["Wind_Speed"]), 2)
        rain = round(float(row["Rainfall"]), 2)
        condition = row["Condition"]

        load = predict_load(weekday, temp, condition, humidity, wind_speed, holiday, event, rain,
                             solar_generation, low_price, high_price, avg_price, qoq_price,
                             day, month, year, day_of_year, hour)

        time_str = f"{hour:02d}-00:{(hour + 1) % 24:02d}:00"

        doc = create_document({
            'Date': date_str, 'Time': time_str, 'Weekday': calendar.day_name[weekday],
            'Temperature': temp, 'Condition': condition, 'Humidity': humidity,
            'Wind_Speed': wind_speed, 'Holiday': holiday, 'Event': event, 'Load': load,
        })
        date_documents.append(doc)

        date_csv_rows.append({
            'Date': date_str, 'Time': time_str, "Weekday": calendar.day_name[weekday],
            "Temperature": temp, "Condition": condition, "Humidity": humidity,
            "Wind_Speed": wind_speed, "Holiday": holiday, "Event": event, "Rainfall": rain,
            "Solar_Generation": round(solar_generation, 2), "low_price": round(low_price, 2),
            "high_price": round(high_price, 2), "Average_Price_Rs_Per_Sqft": round(avg_price, 2),
            "QoQ_Price_Change_Percent": round(qoq_price, 2), 'Load': load,
            "BRPL": None, "BYPL": None, "NDPL": None, "NDMC": None, "MES": None,
        })

    if not date_documents:
        continue

    upsert_date_documents(date_documents)
    successfully_processed_dates.add(date_str)
    print(f"Upserted {len(date_documents)} hours for {date_str} into FutureData.")

    if full_date.date() == tomorrow_date_obj:
        tomorrow_csv_rows = date_csv_rows
        for doc in date_documents:
            db.data.update_one(
                {"Date": doc["Date"], "Time": doc["Time"]},
                {"$set": doc},
                upsert=True,
            )

if tomorrow_csv_rows:
    df_rows = pd.DataFrame(tomorrow_csv_rows, columns=CSV_COLUMNS)
    df_rows.to_csv(csv_file, index=False, mode='a', header=False)
    print(f"Appended {len(df_rows)} rows for tomorrow to All_Data.csv.")
else:
    print("WARNING: No rows built for tomorrow — CSV not updated this run.")

# ---------------------------------------------------------------------------
# Regenerate data.json from the (now updated) All_Data.csv
# ---------------------------------------------------------------------------
DROP_COLUMNS = {
    "Rainfall", "Solar_Generation", "low_price", "high_price",
    "Average_Price_Rs_Per_Sqft", "QoQ_Price_Change_Percent",
    "BRPL", "BYPL", "NDPL", "NDMC", "MES"
}

df_all = pd.read_csv(csv_file, header=None, names=CSV_COLUMNS, encoding="ISO-8859-1", low_memory=False)
df_all = df_all[df_all["Date"] != "Date"].reset_index(drop=True)


def convert_types(row):
    return {
        "Date": row["Date"], "Time": row["Time"], "Weekday": row["Weekday"],
        "Temperature": float(row["Temperature"]) if pd.notna(row["Temperature"]) else None,
        "Condition": row["Condition"],
        "Humidity": int(row["Humidity"]) if pd.notna(row["Humidity"]) else None,
        "Wind_Speed": float(row["Wind_Speed"]) if pd.notna(row["Wind_Speed"]) else None,
        "Holiday": bool(row["Holiday"]) if pd.notna(row["Holiday"]) else False,
        "Event": row["Event"] if pd.notna(row["Event"]) and row["Event"] not in ["No", ""] else None,
        "Load": float(row["Load"]) if pd.notna(row["Load"]) else None,
    }


json_rows = [convert_types(row) for _, row in df_all.drop(columns=list(DROP_COLUMNS), errors="ignore").iterrows()]
with open(json_file, mode="w", encoding="utf-8") as file:
    json.dump(json_rows, file, indent=4)

print(f"\nRegenerated data.json: {len(json_rows)} rows.")
print(f"Processed dates: {sorted(successfully_processed_dates)}")
print("Daily run complete.")

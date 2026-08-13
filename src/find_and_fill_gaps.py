"""
find_and_fill_gaps.py

1. Scans data/All_Data.csv for missing dates/hours between START_DATE and
   yesterday (today's row isn't expected yet — the daily script only writes
   "tomorrow's" row the night before).
2. For any missing date, fetches real hourly weather from Open-Meteo:
     - Recent dates (last ~5 days): forecast API with past_days (no lag)
     - Older dates: archive API (finalized historical data)
3. Builds full feature rows (matching your model's training schema) and
   APPENDS them directly to All_Data.csv in the exact same 20-column format
   your daily script already writes — no schema changes, no extra columns.
4. Skips any (Date, Time) already present — safe to re-run.
5. Dry-run first (prints what would be added); only writes if DRY_RUN=False.

Does NOT touch data.json — run csv_to_json.py / the "Regenerate JSON from CSV"
workflow afterward to sync it.
"""

import os
import sys
import gzip
import pickle
import calendar
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
CSV_PATH = os.path.join("data", "All_Data.csv")
HOLIDAY_PATH = os.path.join("data", "Holidays.csv")
SOLAR_PATH = "solar_data_forecast.csv"
REAL_ESTATE_PATH = "real_estate_price_forecast.csv"
MODEL_PATH = "model.pkl.gz"

START_DATE = "2026-04-21"  # earliest date we care about checking
LATITUDE = 28.6139
LONGITUDE = 77.2090
DRY_RUN = False  # set False only after reviewing the report below

COLUMNS = [
    "Date", "Time", "Weekday", "Temperature", "Condition", "Humidity",
    "Wind_Speed", "Holiday", "Event", "Rainfall", "Solar_Generation",
    "low_price", "high_price", "Average_Price_Rs_Per_Sqft",
    "QoQ_Price_Change_Percent", "Load", "BRPL", "BYPL", "NDPL", "NDMC", "MES"
]

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
# Step 1: Load existing CSV and find what's missing
# ---------------------------------------------------------------------------
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH, header=None, names=COLUMNS, encoding="ISO-8859-1", low_memory=False)
existing_pairs = set(zip(df["Date"], df["Time"]))
existing_dates = set(df["Date"])

today = datetime.now().date()
yesterday = today - timedelta(days=1)
start = datetime.strptime(START_DATE, "%Y-%m-%d").date()

expected_dates = []
d = start
while d <= yesterday:
    expected_dates.append(d)
    d += timedelta(days=1)

missing_dates = []
for d in expected_dates:
    date_str = d.strftime("%d-%m-%Y")
    hours_present = sum(1 for date, time in existing_pairs if date == date_str)
    if hours_present < 24:
        missing_dates.append((d, hours_present))

print(f"--- Gap report: {START_DATE} to {yesterday} ---")
print(f"Total dates checked: {len(expected_dates)}")
print(f"Dates with missing hours: {len(missing_dates)}")
for d, count in missing_dates:
    print(f"  {d.strftime('%d-%m-%Y')}: {count}/24 hours present")

if not missing_dates:
    print("\nNo gaps found. CSV is complete for this range.")
    sys.exit(0)

if DRY_RUN:
    print(f"\nDRY_RUN is True — no data was fetched or written.")
    print(f"Review the gap list above. If correct, set DRY_RUN = False and re-run.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Step 2: Load supporting data + model (only needed if actually filling)
# ---------------------------------------------------------------------------
holiday_data = pd.read_csv(HOLIDAY_PATH)
solar_data = pd.read_csv(SOLAR_PATH)
real_estate_data = pd.read_csv(REAL_ESTATE_PATH)
real_estate_data["date"] = pd.to_datetime(real_estate_data["date"], dayfirst=True)

with gzip.open(MODEL_PATH, "rb") as f:
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
    return np.sin(2 * np.pi * value / max_value), np.cos(2 * np.pi * value / max_value)


def fetch_weather_range(start_date_obj, end_date_obj, use_forecast_api):
    """Fetch a whole contiguous date range in ONE request instead of looping
    day by day — far fewer network calls, far less prone to timeouts."""
    if use_forecast_api:
        days_ago = (datetime.now().date() - start_date_obj).days
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "past_days": max(days_ago, 1),
            "forecast_days": 1,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "Asia/Kolkata",
        }
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "start_date": start_date_obj.strftime("%Y-%m-%d"),
            "end_date": end_date_obj.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "Asia/Kolkata",
        }

    last_error = None
    for attempt in range(3):  # retry up to 3 times on timeout
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()["hourly"]
            wdf = pd.DataFrame({
                "datetime": pd.to_datetime(data["time"]),
                "Temperature": data["temperature_2m"],
                "Humidity": data["relative_humidity_2m"],
                "Rainfall": data["precipitation"],
                "Wind_Speed": data["wind_speed_10m"],
                "weather_code": data["weather_code"],
            })
            wdf["Condition"] = wdf["weather_code"].map(WMO_CONDITION_MAP).fillna("Cloudy")
            return wdf
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt + 1} failed: {e}")
    raise last_error


def group_into_contiguous_ranges(dates):
    """Turn a sorted list of dates into a list of (start, end) contiguous ranges."""
    if not dates:
        return []
    ranges = []
    range_start = dates[0]
    prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days > 1:
            ranges.append((range_start, prev))
            range_start = d
        prev = d
    ranges.append((range_start, prev))
    return ranges


# ---------------------------------------------------------------------------
# Step 3: Build and append rows for each missing date
# ---------------------------------------------------------------------------
RECENT_THRESHOLD_DAYS = 5  # use forecast API (past_days) within this window
new_rows = []
skipped_existing = 0

# Split missing dates into "recent" (forecast API) vs "older" (archive API)
missing_date_objs = sorted([d for d, _ in missing_dates])
today_date = datetime.now().date()

recent_dates = [d for d in missing_date_objs if (today_date - d).days <= RECENT_THRESHOLD_DAYS]
older_dates = [d for d in missing_date_objs if (today_date - d).days > RECENT_THRESHOLD_DAYS]

recent_ranges = group_into_contiguous_ranges(recent_dates)
older_ranges = group_into_contiguous_ranges(older_dates)

all_weather_frames = []

for start, end in older_ranges:
    print(f"\nFetching {start.strftime('%d-%m-%Y')} to {end.strftime('%d-%m-%Y')} via archive API (one request)...")
    try:
        wdf = fetch_weather_range(start, end, use_forecast_api=False)
        all_weather_frames.append(wdf)
        print(f"  Got {len(wdf)} hourly records.")
    except Exception as e:
        print(f"  FAILED after retries: {e}")

for start, end in recent_ranges:
    print(f"\nFetching {start.strftime('%d-%m-%Y')} to {end.strftime('%d-%m-%Y')} via forecast (past_days) API (one request)...")
    try:
        wdf = fetch_weather_range(start, end, use_forecast_api=True)
        # forecast API returns a fixed window — filter to just the dates we need
        wdf = wdf[wdf["datetime"].dt.date.isin(recent_dates)]
        all_weather_frames.append(wdf)
        print(f"  Got {len(wdf)} hourly records.")
    except Exception as e:
        print(f"  FAILED after retries: {e}")

if not all_weather_frames:
    print("\nNo weather data could be fetched. Exiting.")
    sys.exit(1)

combined_weather = pd.concat(all_weather_frames, ignore_index=True)
print(f"\nTotal hourly weather records fetched: {len(combined_weather)}")

for _, row in combined_weather.iterrows():
    dt = row["datetime"]
    date_str = dt.strftime("%d-%m-%Y")
    time_str = f"{dt.hour:02d}-00:{(dt.hour + 1) % 24:02d}:00"

    if (date_str, time_str) in existing_pairs:
        skipped_existing += 1
        continue

    day, month, year, hour = dt.day, dt.month, dt.year, dt.hour
    weekday = dt.weekday()
    day_of_year = dt.timetuple().tm_yday

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

    month_start = f"{year}-{month:02d}-01"
    monthly_solar = solar_data[solar_data["Date"] == month_start]
    if monthly_solar.empty:
        continue
    last_day = calendar.monthrange(year, month)[1]
    solar_generation = round(monthly_solar["Forecasted Solar Generation"].values[0], 2) / last_day

    date_ts = pd.to_datetime(f"{year}-{month}-{day}")
    quarter_mask = (real_estate_data["date"].dt.year == year) & (
        real_estate_data["date"].dt.quarter == date_ts.quarter
    )
    qre = real_estate_data[quarter_mask]
    if qre.empty:
        continue
    low_price = qre["low_price_pred"].values.item()
    high_price = qre["high_price_pred"].values.item()
    avg_price = qre["Average_Price"].values.item()
    qoq_price = qre["QoQ_Price_Change_Percent"].values.item()

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

    new_rows.append({
        "Date": date_str, "Time": time_str, "Weekday": calendar.day_name[weekday],
        "Temperature": temp, "Condition": condition, "Humidity": humidity,
        "Wind_Speed": wind_speed, "Holiday": holiday, "Event": event,
        "Rainfall": rain, "Solar_Generation": round(solar_generation, 2),
        "low_price": round(low_price, 2), "high_price": round(high_price, 2),
        "Average_Price_Rs_Per_Sqft": round(avg_price, 2),
        "QoQ_Price_Change_Percent": round(qoq_price, 2), "Load": load,
        "BRPL": None, "BYPL": None, "NDPL": None, "NDMC": None, "MES": None,
    })

print(f"\nBuilt {len(new_rows)} new rows (skipped {skipped_existing} already-present hours).")

if new_rows:
    new_df = pd.DataFrame(new_rows, columns=COLUMNS)
    new_df.to_csv(CSV_PATH, index=False, mode="a", header=False, encoding="ISO-8859-1")
    print(f"Appended {len(new_rows)} rows to {CSV_PATH}.")
else:
    print("Nothing new to append.")

print("\nDone. Run csv_to_json.py (or the 'Regenerate JSON from CSV' workflow) next to sync data.json.")

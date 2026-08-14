import sys
import time
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from datetime import datetime, timedelta
import calendar
import pandas as pd
import numpy as np
import gzip
import pickle
import json
import os
import pymongo
import undetected_chromedriver as uc
from dotenv import load_dotenv

load_dotenv()

LATITUDE = 28.6139
LONGITUDE = 77.2090

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


import subprocess
import re


def get_chrome_major_version():
    """Detect the exact major version of Chrome actually installed on this
    runner, so we can force undetected_chromedriver to fetch a matching
    chromedriver build instead of letting it guess (which has been wrong
    twice now when Chrome auto-updates)."""
    try:
        output = subprocess.check_output(["/usr/bin/google-chrome", "--version"]).decode()
        match = re.search(r"(\d+)\.", output)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"Could not determine Chrome version: {e}")
    return None


def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    print("Chrome binary exists at /usr/bin/google-chrome:", os.path.exists("/usr/bin/google-chrome"))
    chrome_major = get_chrome_major_version()
    print(f"Detected installed Chrome major version: {chrome_major}")

    last_error = None
    for attempt in range(2):
        try:
            driver = uc.Chrome(
                options=options,
                headless=True,
                use_subprocess=True,
                browser_executable_path="/usr/bin/google-chrome",
                version_main=chrome_major,
            )
            return driver
        except Exception as e:
            last_error = e
            print(f"Chrome init attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise last_error


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

# NOTE: No upfront wipe of the whole collection anymore. Each date's
# documents only get replaced once we've actually got fresh data for that
# specific date. A date that fails to scrape today keeps whatever it had
# before, instead of losing its data entirely.


def replace_date_documents(date_str, documents):
    collection.delete_many({"Date": date_str})
    if documents:
        collection.insert_many(documents)


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


holiday_data_path = os.path.join("data", "Holidays.csv")
holiday_data = pd.read_csv(holiday_data_path)
solar_data_path = os.path.join("solar_data_forecast.csv")
solar_data = pd.read_csv(solar_data_path)
real_estate_data_path = os.path.join("real_estate_price_forecast.csv")
real_estate_data = pd.read_csv(real_estate_data_path)
real_estate_data['date'] = pd.to_datetime(real_estate_data['date'], dayfirst=True)
with gzip.open('model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

csv_file = os.path.join("data", "All_Data.csv")
json_file = os.path.join("data", "data.json")
file_path = os.path.join("data", "Forecast_Data.csv")

if os.path.exists(file_path):
    os.remove(file_path)


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
    month_start = f"{year}-{month:02d}-01"
    monthly_solar_data = solar_data[solar_data['Date'] == month_start]
    if monthly_solar_data.empty:
        return None
    last_day = calendar.monthrange(year, month)[1]
    return round(monthly_solar_data['Forecasted Solar Generation'].values[0], 2) / last_day


def get_real_estate(year, date):
    quarter_mask = (real_estate_data['date'].dt.year == year) & (real_estate_data['date'].dt.quarter == date.quarter)
    q = real_estate_data[quarter_mask]
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


driver = init_driver()
driver.get("https://www.timeanddate.com/weather/india/new-delhi/hourly")
time.sleep(3)

print(f"Page title: {driver.title}")
print(f"Page source snippet: {driver.page_source[:1000]}")

elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/weather/india/new-delhi/hourly?hd=')]")
print(f"Found {len(elements)} date-link elements")

date_links = []
for elem in elements:
    href = elem.get_attribute("href")
    hd_param = href.split("hd=")[-1]
    date_links.append(hd_param)

print(f"date_links: {date_links}")

if not date_links:
    print("WARNING: No date links found — site may be blocking automated access. Exiting without changes.")
    driver.quit()
    sys.exit(1)

successfully_scraped_dates = set()

for hd in date_links:
    url = f"https://www.timeanddate.com/weather/india/new-delhi/hourly?hd={hd}"
    driver.get(url)

    try:
        table = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "wt-hbh")))
    except TimeoutException:
        print(f"Table not found for {hd} after explicit wait — retrying once...")
        time.sleep(3)
        try:
            table = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "wt-hbh")))
        except TimeoutException:
            print(f"Table still not found for {hd} after retry — skipping (existing data, if any, preserved).")
            continue

    year = int(hd[:4])
    day = int(hd[6:8])
    month = int(hd[4:6])
    full_date = datetime(year, month, day)
    weekday = full_date.weekday()
    day_of_year = full_date.timetuple().tm_yday
    date_str = f"{day:02d}-{month:02d}-{year}"

    rows = table.find_elements(By.TAG_NAME, "tr")[2:]
    date_documents = []
    date_data_rows = []

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "th") + row.find_elements(By.TAG_NAME, "td")
        if len(cols) < 10:
            continue

        time_cell = cols[0].text.strip()
        hour_part = time_cell.split("\n", 1)[0] if "\n" in time_cell else time_cell

        if 'pm' in hour_part and '12:' not in hour_part:
            hour = int(hour_part.split(":")[0]) + 12
        elif 'am' in hour_part and '12:' in hour_part:
            hour = 0
        else:
            hour = int(hour_part.split(":")[0])

        temp_text = cols[2].text.strip()
        if '°F' in temp_text:
            temp = (int(temp_text.replace("°F", "").strip()) - 32) * 5 / 9
        elif '°C' in temp_text:
            temp = int(temp_text.replace("°C", "").strip())
        else:
            continue

        condition = cols[3].text.strip().rstrip('.')

        def parse_measurement(text, units):
            for unit in units:
                if unit in text:
                    value = text.replace(unit, "").strip()
                    try:
                        return float(value)
                    except ValueError:
                        return None
            return None

        wind_speed = parse_measurement(cols[5].text.strip(), ["km/h", "mph"])
        wind_speed = round(wind_speed, 2) if wind_speed is not None else 0

        humidity = int(cols[7].text.replace("%", "").strip())
        holiday, event = get_holiday_event(day, month, year, weekday)

        rain_text = cols[9].text.replace('mm (rain)', '').strip()
        rain = float(rain_text) if rain_text.replace('.', '', 1).isdigit() else 0.0

        date_ts = pd.to_datetime(f"{year}-{month}-{day}")
        solar_generation = get_solar_generation(year, month)
        real_estate = get_real_estate(year, date_ts)
        if solar_generation is None or real_estate is None:
            continue
        low_price, high_price, avg_price, qoq_price = real_estate

        load = predict_load(weekday, temp, condition, humidity, wind_speed, holiday, event, rain,
                             solar_generation, low_price, high_price, avg_price, qoq_price,
                             day, month, year, day_of_year, hour)

        hour_24 = hour % 24
        hour_new = f"{hour_24:02d}"
        hour_next = f"{(hour_24 + 1) % 24:02d}"
        time_str = f"{hour_new}-00:{hour_next}:00"

        doc = create_document({
            'Date': date_str, 'Time': time_str, 'Weekday': calendar.day_name[weekday],
            'Temperature': round(temp, 2), 'Condition': condition, 'Humidity': humidity,
            'Wind_Speed': wind_speed, 'Holiday': holiday, 'Event': event, 'Load': load,
        })
        date_documents.append(doc)

        date_data_rows.append({
            'Date': date_str, 'Time': time_str, "Weekday": calendar.day_name[weekday],
            "Temperature": round(temp, 2), "Condition": condition, "Humidity": humidity,
            "Wind_Speed": wind_speed, "Holiday": holiday, "Event": event, "Rainfall": rain,
            "Solar_Generation": round(solar_generation, 2), "low_price": round(low_price, 2),
            "high_price": round(high_price, 2), "Average_Price_Rs_Per_Sqft": round(avg_price, 2),
            "QoQ_Price_Change_Percent": round(qoq_price, 2), 'Load': load,
            "BRPL": None, "BYPL": None, "NDPL": None, "NDMC": None, "MES": None,
        })

    if not date_documents:
        print(f"No usable rows extracted for {hd} — skipping (existing data, if any, preserved).")
        continue

    replace_date_documents(date_str, date_documents)
    successfully_scraped_dates.add(date_str)
    print(f"Upserted {len(date_documents)} hours for {date_str} into FutureData.")

    today = datetime.now()
    is_tomorrow = (day == today.day + 1 and month == today.month and year == today.year)

    if is_tomorrow:
        df_rows = pd.DataFrame(date_data_rows, columns=[
            'Date', 'Time', "Weekday", "Temperature", "Condition", "Humidity", "Wind_Speed", "Holiday", "Event",
            "Rainfall", "Solar_Generation", "low_price", "high_price", "Average_Price_Rs_Per_Sqft",
            "QoQ_Price_Change_Percent", 'Load', 'BRPL', 'BYPL', 'NDPL', 'NDMC', 'MES'
        ])
        df_rows.to_csv(csv_file, index=False, mode='a', header=False)
        print(f"Appended {len(df_rows)} rows for tomorrow ({date_str}) to All_Data.csv.")

        db.data.delete_many({"Date": date_str})
        db.data.insert_many(date_documents)

driver.quit()

# ---------------------------------------------------------------------------
# Guaranteed fallback: make sure TOMORROW is always complete
# ---------------------------------------------------------------------------
today = datetime.now()
tomorrow = today + timedelta(days=1)
tomorrow_date_str = f"{tomorrow.day:02d}-{tomorrow.month:02d}-{tomorrow.year}"
tomorrow_count = collection.count_documents({"Date": tomorrow_date_str})
print(f"\nTomorrow ({tomorrow_date_str}) has {tomorrow_count}/24 hours in FutureData after scraping.")

if tomorrow_count < 24:
    print("Tomorrow is incomplete — falling back to Open-Meteo forecast API to fill it in.")
    try:
        om_url = "https://api.open-meteo.com/v1/forecast"
        om_params = {
            "latitude": LATITUDE, "longitude": LONGITUDE, "forecast_days": 2,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(om_url, params=om_params, timeout=60)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]

        om_df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"]),
            "Temperature": hourly["temperature_2m"], "Humidity": hourly["relative_humidity_2m"],
            "Rainfall": hourly["precipitation"], "Wind_Speed": hourly["wind_speed_10m"],
            "weather_code": hourly["weather_code"],
        })
        om_df["Condition"] = om_df["weather_code"].map(WMO_CONDITION_MAP).fillna("Cloudy")
        om_df = om_df[om_df["datetime"].dt.date == tomorrow.date()]

        fallback_docs = []
        fallback_csv_rows = []

        for _, row in om_df.iterrows():
            dt = row["datetime"]
            day, month, year, hour = dt.day, dt.month, dt.year, dt.hour
            weekday = dt.weekday()
            day_of_year = dt.timetuple().tm_yday
            date_str = f"{day:02d}-{month:02d}-{year}"
            time_str = f"{hour:02d}-00:{(hour + 1) % 24:02d}:00"

            holiday, event = get_holiday_event(day, month, year, weekday)
            solar_generation = get_solar_generation(year, month)
            date_ts = pd.to_datetime(f"{year}-{month}-{day}")
            real_estate = get_real_estate(year, date_ts)
            if solar_generation is None or real_estate is None:
                continue
            low_price, high_price, avg_price, qoq_price = real_estate

            temp = round(float(row["Temperature"]), 2)
            humidity = int(round(row["Humidity"]))
            wind_speed = round(float(row["Wind_Speed"]), 2)
            rain = round(float(row["Rainfall"]), 2)
            condition = row["Condition"]

            load = predict_load(weekday, temp, condition, humidity, wind_speed, holiday, event, rain,
                                 solar_generation, low_price, high_price, avg_price, qoq_price,
                                 day, month, year, day_of_year, hour)

            doc = create_document({
                'Date': date_str, 'Time': time_str, 'Weekday': calendar.day_name[weekday],
                'Temperature': temp, 'Condition': condition, 'Humidity': humidity,
                'Wind_Speed': wind_speed, 'Holiday': holiday, 'Event': event, 'Load': load,
            })
            fallback_docs.append(doc)

            fallback_csv_rows.append({
                'Date': date_str, 'Time': time_str, "Weekday": calendar.day_name[weekday],
                "Temperature": temp, "Condition": condition, "Humidity": humidity,
                "Wind_Speed": wind_speed, "Holiday": holiday, "Event": event, "Rainfall": rain,
                "Solar_Generation": round(solar_generation, 2), "low_price": round(low_price, 2),
                "high_price": round(high_price, 2), "Average_Price_Rs_Per_Sqft": round(avg_price, 2),
                "QoQ_Price_Change_Percent": round(qoq_price, 2), 'Load': load,
                "BRPL": None, "BYPL": None, "NDPL": None, "NDMC": None, "MES": None,
            })

        if fallback_docs:
            replace_date_documents(tomorrow_date_str, fallback_docs)
            db.data.delete_many({"Date": tomorrow_date_str})
            db.data.insert_many(fallback_docs)
            print(f"Fallback: filled {len(fallback_docs)} hours for {tomorrow_date_str} via Open-Meteo.")

            if tomorrow_date_str not in successfully_scraped_dates:
                fallback_df = pd.DataFrame(fallback_csv_rows, columns=[
                    'Date', 'Time', "Weekday", "Temperature", "Condition", "Humidity", "Wind_Speed", "Holiday", "Event",
                    "Rainfall", "Solar_Generation", "low_price", "high_price", "Average_Price_Rs_Per_Sqft",
                    "QoQ_Price_Change_Percent", 'Load', 'BRPL', 'BYPL', 'NDPL', 'NDMC', 'MES'
                ])
                fallback_df.to_csv(csv_file, index=False, mode='a', header=False)
                print(f"Fallback: appended {len(fallback_df)} rows for {tomorrow_date_str} to All_Data.csv.")
        else:
            print("Fallback: Open-Meteo returned no usable rows either — tomorrow remains incomplete.")
    except Exception as e:
        print(f"Fallback to Open-Meteo failed: {str(e)}")
else:
    print("Tomorrow is already complete from scraping — no fallback needed.")

# ---------------------------------------------------------------------------
# Regenerate data.json from the (now updated) All_Data.csv
# ---------------------------------------------------------------------------
COLUMNS = [
    "Date", "Time", "Weekday", "Temperature", "Condition", "Humidity",
    "Wind_Speed", "Holiday", "Event", "Rainfall", "Solar_Generation",
    "low_price", "high_price", "Average_Price_Rs_Per_Sqft",
    "QoQ_Price_Change_Percent", "Load", "BRPL", "BYPL", "NDPL", "NDMC", "MES"
]
DROP_COLUMNS = {
    "Rainfall", "Solar_Generation", "low_price", "high_price",
    "Average_Price_Rs_Per_Sqft", "QoQ_Price_Change_Percent",
    "BRPL", "BYPL", "NDPL", "NDMC", "MES"
}

df_all = pd.read_csv(csv_file, header=None, names=COLUMNS, encoding="ISO-8859-1", low_memory=False)
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
print("Daily run complete.")

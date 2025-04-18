import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from datetime import datetime
import calendar
import pandas as pd
import numpy as np
import gzip
import pickle
import csv
import json
import os
import pymongo
from dotenv import load_dotenv

load_dotenv()

def init_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    service = Service("/usr/local/bin/chromedriver")
    return webdriver.Chrome(
        service=service,
        options=options
    )

# MongoDB connection setup
try:
    # Verify MongoDB URI is loaded
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables")
    
    print(f"Connecting to MongoDB at: {mongodb_uri[:20]}...")  # Log partial URI
    
    client = pymongo.MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
    
    # Test connection
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
today_date = datetime.now().strftime("%d-%m-%Y")
today_data = list(collection.find({"Date": today_date}))
collection.delete_many({})
if today_data:
    collection.insert_many(today_data)

def create_document(data_row):
    """Create MongoDB document from data row"""
    doc = {
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
    
    return doc

# Initialize WebDriver with headless mode
driver = init_driver()
driver.get("https://www.timeanddate.com/weather/india/new-delhi/hourly")
driver.implicitly_wait(10)

holiday_data_path = os.path.join("data", "Holidays.csv")
holiday_data = pd.read_csv(holiday_data_path)
solar_data_path = os.path.join("solar_data_forecast.csv")
solar_data = pd.read_csv(solar_data_path)
real_estate_data_path = os.path.join("real_estate_price_forecast.csv")
real_estate_data = pd.read_csv(real_estate_data_path)
model_path = os.path.join("model.pkl")
with gzip.open('model.pkl.gz', 'rb') as f:
    model = pickle.load(f)
csv_file = os.path.join("data", "All_Data.csv")
json_file = os.path.join("data", "data.json")
file_path = os.path.join("data", "Forecast_Data.csv")

if os.path.exists(file_path):
    os.remove(file_path)

# Collect date parameters from hrefs
date_links = []
elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/weather/india/new-delhi/hourly?hd=')]")
for elem in elements:
    href = elem.get_attribute("href")
    # Extract unique date parameter (e.g., "hd=20250407")
    hd_param = href.split("hd=")[-1]
    date_links.append(hd_param)

weather_data = []

for hd in date_links:
    # Navigate directly using the date parameter
    url = f"https://www.timeanddate.com/weather/india/new-delhi/hourly?hd={hd}"
    driver.get(url)
    
    try:
        table = driver.find_element(By.ID, "wt-hbh")
    except:
        print(f"Table not found for {hd}")
        continue

    # Extract year from the page title
    year = int(hd[:4])  # Extract year from hd_param (format: YYYYMMDD)
    
    rows = table.find_elements(By.TAG_NAME, "tr")[2:]  # Skip header rows
    
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "th") + row.find_elements(By.TAG_NAME, "td")
        if len(cols) < 10:
            continue
        
        # Extract time and date
        time_cell = cols[0].text.strip()
        if "\n" in time_cell:
            hour_part, date_part = time_cell.split("\n", 1)
        else:
            hour_part = time_cell
            date_part = None
            
        # Convert 12-hour format into 24-hour
        if 'pm' in hour_part and '12:' not in hour_part:
            hour = int(hour_part.split(":")[0]) + 12
        elif 'am' in hour_part and '12:' in hour_part:
            hour = 0  # Midnight
        else:
            hour = int(hour_part.split(":")[0])
        
        # Parse date using hd_param (YYYYMMDD)
        day = int(hd[6:8])
        month = int(hd[4:6])
        full_date = datetime(year, month, day)
        weekday = full_date.weekday()  # Monday = 0
        day_of_year = full_date.timetuple().tm_yday
        
        # Temperature parsing with unit conversion
        temp_text = cols[2].text.strip()  # Verify correct column index
        if '°F' in temp_text:
            temp_f = int(temp_text.replace("°F", "").strip())
            temp = (temp_f - 32) * 5/9  # Convert Fahrenheit to Celsius
        elif '°C' in temp_text:
            temp = int(temp_text.replace("°C", "").strip())
        
        # Weather condition
        condition = cols[3].text.strip().rstrip('.')
        
        # Unified unit parsing function
        def parse_measurement(text, units):
            """Parse measurements with automatic unit conversion"""
            for unit in units:
                if unit in text:
                    value = text.replace(unit, "").strip()
                    try:
                        num = float(value)
                        return num
                    except ValueError:
                        return None
            return None

        # Wind speed parsing with conversion
        wind_speed_text = cols[5].text.strip()
        wind_speed = parse_measurement(wind_speed_text, ["km/h", "mph"])

        if wind_speed is not None:
            wind_speed = round(wind_speed, 2)
        else:
            wind_speed = 0  # Default value or skip the record
        
        # Humidity (%)
        humidity = int(cols[7].text.replace("%", "").strip())
        
        # Function to remove duplicate words while preserving order
        def unique_event_concat(events):
            words = "/".join(events).split("/")  # Flatten the list into words
            unique_words = []  
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)  # Keep only unique words
            return "/".join(unique_words)

        # Group by Day, Month, Year and apply custom concatenation
        holiday_data_grouped = holiday_data.groupby(['Day', 'Month', 'Year'], as_index=False).agg({
            'Holiday': 'first',
            'Event': unique_event_concat  # Remove duplicates in the concatenated event names
        })

        matched_row = holiday_data_grouped[
            (holiday_data_grouped['Day'] == day) &
            (holiday_data_grouped['Month'] == month) &
            (holiday_data_grouped['Year'] == year)
        ]

        if not matched_row.empty:
            holiday = matched_row['Holiday'].values[0]  # Extract the value
            event = matched_row['Event'].values[0]  # Extract the event
        else:
            holiday = 0
            event = 'No'
            if weekday in [5, 6]: 
                holiday = 1
                event = 'Weekend'
        
        # Extract text, clean it, and handle missing values
        rain_text = cols[9].text.replace('mm (rain)', '').strip()

        # Convert to float, or set to 0 if missing ('-')
        rain = float(rain_text) if rain_text.replace('.', '', 1).isdigit() else 0.0
        
        # Convert date to Pandas datetime
        date = pd.to_datetime(f"{year}-{month}-{day}")
        
        # Get the first day of the month in YYYY-MM-01 format
        month_start = f"{year}-{month:02d}-01"

        # Filter data for the given month
        monthly_solar_data = solar_data[solar_data['Date'] == month_start]

        # Get last day of the month
        last_day = calendar.monthrange(year, month)[1]

        solar_generation = round(monthly_solar_data['Forecasted Solar Generation'].values[0], 2) / last_day
        
        # Get quarter start month
        quarter_start_month = (date.quarter - 1) * 3 + 1  # Q1 → Jan, Q2 → Apr, etc.
        quarter_start = f"{date.year}-{quarter_start_month:02d}-01"

        real_estate_data['date'] = pd.to_datetime(real_estate_data['date'], dayfirst=True)
        # Filter data for the quarter
        quarter_mask = (real_estate_data['date'].dt.year == year) & (real_estate_data['date'].dt.quarter == date.quarter)
        quarter_real_estate_data = real_estate_data[quarter_mask]

        low_price = quarter_real_estate_data['low_price_pred'].values.item()
        high_price = quarter_real_estate_data['high_price_pred'].values.item()
        avg_price = quarter_real_estate_data['Average_Price'].values.item()
        QoQ_price = quarter_real_estate_data['QoQ_Price_Change_Percent'].values.item()
        
        # Function to compute sin and cos transformations
        def cyclic_encoding(value, max_value):
            sin_value = np.sin(2 * np.pi * value / max_value)
            cos_value = np.cos(2 * np.pi * value / max_value)
            return sin_value, cos_value

        # Compute cyclic encodings
        hour_sin, hour_cos = cyclic_encoding(hour, 24)
        weekday_sin, weekday_cos = cyclic_encoding(weekday, 7)
        month_sin, month_cos = cyclic_encoding(month, 12)
        dayofyear_sin, dayofyear_cos = cyclic_encoding(day_of_year, 365)

        # Interaction Feature
        temp_x_hour = round(temp, 2) * hour
        
        prediction = model.predict(pd.DataFrame({"Weekday": [weekday], "Temperature": [round(temp, 2)], "Condition": [condition], "Humidity": [humidity], 
                    "Wind_Speed": [wind_speed], "Holiday": [holiday], "Event": [event], "Rainfall": [rain], 
                    "Solar_Generation": [round(solar_generation, 2)], "low_price": [round(low_price, 2)], 
                    "high_price": [round(high_price, 2)], "Average_Price_Rs_Per_Sqft": [round(avg_price, 2)], 
                    "QoQ_Price_Change_Percent": [round(QoQ_price, 2)], "Day": [day], "Month": [month], "Year": [year], 
                    "DayOfYear": [day_of_year], "Hour": [hour], "Hour_sin": [hour_sin], "Hour_cos": [hour_cos], 
                    "Weekday_sin": [weekday_sin], "Weekday_cos": [weekday_cos], "Month_sin": [month_sin], 
                    "Month_cos": [month_cos], "DayOfYear_sin": [dayofyear_sin], "DayOfYear_cos": [dayofyear_cos],
                    "temp_x_hour": [temp_x_hour]}))
        
        today = datetime.now().strftime("%d-%m-%Y")
        current_day, current_month, current_year = map(int, today.split('-'))
        
        day_new = str(day).zfill(2)
        month_new = str(month).zfill(2)
        # Handle 24-hour conversion for output formatting
        hour_24 = hour % 24  # Ensure 23 becomes 23, 24 becomes 0
        hour_new = f"{hour_24:02d}"
        hour_next = f"{(hour_24 + 1) % 24:02d}"
        
        today = datetime.now()
        is_current = (day == (today.day)+1 and month == today.month and year == today.year)
        
        # Create document for MongoDB
        document = create_document({
            'Date': f"{day:02d}-{month:02d}-{year}",
            'Time': f"{hour_new}-00:{hour_next}:00",
            'Weekday': calendar.day_name[weekday],
            'Temperature': round(temp, 2),
            'Condition': condition,
            'Humidity': humidity,
            'Wind_Speed': wind_speed,
            'Holiday': holiday,
            'Event': event,
            'Load': np.round(prediction, 3)[0]
        })

        # Insert into MongoDB
        try:
            # Always insert into FutureData
            db.FutureData.insert_one(document)
            
            # Insert into AllData if current
            if is_current:
                db.data.insert_one(document)
        except Exception as e:
            print(f"MongoDB insertion error: {str(e)}")

        
        all_data_file = "data/All_Data.csv"
        forecast_data_file = "data/Forecast_Data.csv"

        data = {
            'Date': [f'{day_new}-{month_new}-{year}'],
            'Time': [f'{hour_new}-00:{hour_next}:00'],
            "Weekday": [calendar.day_name[weekday]], "Temperature": [round(temp, 2)], "Condition": [condition], "Humidity": [humidity], 
            "Wind_Speed": [wind_speed], "Holiday": [holiday], "Event": [event], "Rainfall": [rain], 
            "Solar_Generation": [round(solar_generation, 2)], "low_price": [round(low_price, 2)], 
            "high_price": [round(high_price, 2)], "Average_Price_Rs_Per_Sqft": [round(avg_price, 2)], 
            "QoQ_Price_Change_Percent": [round(QoQ_price, 2)], 'Load': np.round(prediction, 3),
            "BRPL": None, "BYPL": None, "NDPL": None, "NDMC": None, "MES": None
        }

        data_df = pd.DataFrame(data, columns=[
            'Date', 'Time', "Weekday", "Temperature", "Condition", "Humidity", "Wind_Speed", "Holiday", "Event",
            "Rainfall", "Solar_Generation", "low_price", "high_price", "Average_Price_Rs_Per_Sqft", 
            "QoQ_Price_Change_Percent", 'Load', 'BRPL', 'BYPL', 'NDPL', 'NDMC', 'MES'
        ])

        if day == current_day+1 and month == current_month and year == current_year:
            data_df.to_csv(all_data_file, index=False, mode='a', header=False)

        data_df.to_csv(forecast_data_file, index=False, mode='a', header=not os.path.exists(forecast_data_file))

        weather_data.append([
            weekday, round(temp, 2), condition, humidity, wind_speed, holiday, event, rain, round(solar_generation, 2),
            round(low_price, 2), round(high_price, 2), round(avg_price, 2), round(QoQ_price, 2), day, month, year, day_of_year, hour,
            hour_sin, hour_cos, weekday_sin, weekday_cos, month_sin, month_cos, dayofyear_sin, dayofyear_cos, temp_x_hour
        ])

driver.quit()  # Close browser after all processing

columns = [
    "Weekday", "Temperature", "Condition", "Humidity", "Wind_Speed", "Holiday", "Event",
    "Rainfall", "Solar_Generation", "low_price", "high_price", "Average_Price_Rs_Per_Sqft", "QoQ_Price_Change_Percent",
    "Day", "Month", "Year", "DayOfYear", "Hour", "Hour_sin", "Hour_cos", "Weekday_sin", "Weekday_cos",
    "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos", "temp_x_hour"
]

df = pd.DataFrame(weather_data, columns=columns)

df.to_csv("sample_data.csv", index=False)

drop_columns = {
    "Rainfall", "Solar_Generation", "low_price", "high_price", 
    "Average_Price_Rs_Per_Sqft", "QoQ_Price_Change_Percent", 
    "BRPL", "BYPL", "NDPL", "NDMC", "MES"
}

# Function to convert data types for MongoDB
def convert_types(row):
    return {
        "Date": row["Date"],  # Keep as string
        "Time": row["Time"],  # Keep as string "HH:MM"
        "Weekday": row["Weekday"],  # Keep as string
        "Temperature": float(row["Temperature"]) if row["Temperature"] else None,  # Convert to float
        "Condition": row["Condition"],  # Keep as string
        "Humidity": int(row["Humidity"]) if row["Humidity"] else None,  # Convert to integer
        "Wind_Speed": float(row["Wind_Speed"]) if row["Wind_Speed"] else None,  # Convert to float
        "Holiday": row["Holiday"].strip().lower() == "true",  # Convert "true"/"false" to boolean
        "Event": row["Event"] if row["Event"] else None,  # Keep as string, convert empty to None
        "Load": float(row["Load"]) if row["Load"] else None  # Convert to float
    }

# Read CSV and process data
with open(csv_file, mode="r", encoding="ISO-8859-1") as file:
    csv_reader = csv.DictReader(file)
    data = [convert_types({key: value for key, value in row.items() if key not in drop_columns}) for row in csv_reader]

# Write to JSON file
with open(json_file, mode="w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)
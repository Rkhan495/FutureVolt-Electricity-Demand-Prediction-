
# FutureVolt Electricity Demand Prediction of Delhi

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-orange)
![Time Series](https://img.shields.io/badge/Time%20Series-Forecasting-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

A machine learning system to forecast hourly electricity load demand in Delhi using historical consumption patterns, weather data, calendar events, solar generation, and real estate trends.

---

## 🎯 Objectives

- Forecast hourly electricity demand for Delhi.
- Analyze the effect of features like temperature, humidity, rainfall, holidays, solar generation, and real estate growth.
- Deploy a real-time prediction system that runs automatically via GitHub Actions.

---

## 📌 Features

- **Time-Series Forecasting**: Predicts electricity demand at hourly intervals.
- **Multi-Factor Analysis**: Considers weather, solar, real estate, and events.
- **Real-time Execution**: Automated predictions with daily scheduling.
- **Feature Engineering**: Includes temporal, weather, and event-based features.

---

## 🧩 Project Architecture

This project was collaboratively developed:

- **Prediction Logic (Python)**: Developed by **[Rizwan Khan](https://github.com/Rkhan495)** to process data and generate daily forecasts.
- **Frontend (UI)**: Built and hosted by **[Sanket Bhanuse](https://github.com/SanketBhanuse)** on **[Vercel](https://futurevolt.vercel.app/futuredata)**.
- **Backend Deployment**: Hosted on **[Render](https://futurevolt-backend.onrender.com/api/load/12-04-2025)** by **[Sanket Bhanuse](https://github.com/SanketBhanuse)**.
- **Database**: Shared **MongoDB** instance receives forecasted data.
- **Automation**: A custom GitHub Actions workflow (`daily-prediction.yml`) schedules and executes daily predictions automatically.

> ⚠️ The prediction code is designed to run **only within GitHub Actions**, not locally.

---

##  🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Rkhan495/FutureVolt-Electricity-Demand-Prediction-.git
cd FutureVolt-Electricity-Demand-Prediction-
```

2. Set up Environment and Install dependencies:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```
> ℹ️ requirements.txt includes only dependencies used in the prediction script. Not entire project's packages.

---

## 🗂️ Dataset Structure

The model uses time-series data with the following features:
- Date (DD-MM-YYYY)
- Time (HH:MM-HH:MM)
- Weekday
- Temperature (°C)
- Weather Condition
- Humidity (%)
- Wind Speed (km/h)
- Holiday Flag
- Special Events
- Rainfall (mm)
- Solar Generation (MW)
- Real Estate Data (Low, High, Avg Price/sqft, QoQ %)
- Historical Load (MW)

---

## 🤖 Machine Learning Models

- **Random Forest Regressor**
- **SARIMAX**
- **Facebook Prophet**

---

## ⚙️ GitHub Actions Automation

- Runs at scheduled intervals via GitHub Actions (.github/workflows/daily-prediction.yml).
- Executes prediction scripts.
- Pushes results to the MongoDB database.
- Designed for continuous integration and automatic daily execution.

---

## 🛠️ Tools & Technologies

- Languages: Python
- Libraries: Pandas, NumPy, Scikit-learn, RandomForestRegressor, Prophet, Statsmodels (SARIMAX)
- Automation: GitHub Actions
- Data Collection: Selenium (Web scraping)
- Storage: MongoDB
- Visualization: Matplotlib
- Deployment: Render (backend), Vercel (frontend)

---

## 👨‍💻 Author 
**Rizwan Khan**

**📧 rizwan495khan123@gmail.com**

**🔗 [LinkedIn](https://www.linkedin.com/in/your-linkedin-username)  
🐙 [GitHub](https://github.com/your-github-username)**

---

## 👥 Contributors
- **[Rizwan Khan](https://github.com/Rkhan495)** – Prediction Engine, MongoDB Integration, GitHub Actions Automation
- **[Sanket Bhanuse](https://github.com/SanketBhanuse)** – Frontend UI, Backend Hosting (Render), Deployment (Vercel)

---

### 🔗 Project Repositories

- 🧠 [Prediction Module (This Repo)](https://github.com/Rkhan495/FutureVolt-Electricity-Demand-Prediction-)
- 🎨 [Frontend Repository](https://github.com/SanketBhanuse/FutureVolt)
- ⚙️ [Backend Repository](https://github.com/SanketBhanuse/FutureVolt-Backend)

---

## 📄 License

This project is licensed under the MIT License.

---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

def init_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    return webdriver.Chrome(
        options=options
    )

# Initialize WebDriver with headless mode
driver = init_driver()
print("Chrome version:", driver.capabilities['browserVersion'])
print("ChromeDriver version:", driver.capabilities['chrome']['chromedriverVersion'])
import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

DOWNLOAD_BASE_PATH = Path("downloads")
if not DOWNLOAD_BASE_PATH.exists():
    DOWNLOAD_BASE_PATH.mkdir(parents=True, exist_ok=True)

def download_pre_open_market_data():
    """
    Automates the download of the pre-open market CSV from the NSE India website.
    The file will be saved in the same directory where the script is run.
    """
    # --- 1. Configuration ---

    # The URL of the page with the data
    url = "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market#"

    # Set the download path to the current working directory (the root folder of your project)
    download_path = os.getcwd()
    print(f"Files will be downloaded to: {download_path}")

    # Configure Chrome options
    chrome_options = webdriver.ChromeOptions()
    
    # Set preferences for downloading files
    # - download.default_directory: Specifies the folder to save files in.
    # - download.prompt_for_download: Disables the "Save As" dialog.
    # - safebrowsing.enabled: Recommended to keep enabled.
    prefs = {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # IMPORTANT: Set a realistic User-Agent. NSE website can block default Selenium User-Agents.
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")

    # To run Chrome in the background (without a UI), uncomment the following line
    # chrome_options.add_argument("--headless")

    # --- 2. Initialize WebDriver ---
    
    # Using Selenium's automatic WebDriver manager
    service = ChromeService()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Define a generous timeout for waiting for elements
    wait = WebDriverWait(driver, 20)

    try:
        # --- 3. Navigate and Find the Element ---
        print(f"Navigating to: {url}")
        driver.get(url)

        # The target element is a div with two classes: "downloads" and "downloadLink"
        # We use a CSS Selector for this: 'div.downloads.downloadLink'
        # We wait until the element is present and clickable to avoid errors
        print("Waiting for the download button to be clickable...")
        download_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.downloads.downloadLink"))
        )
        print("Download button found!")

        # --- 4. Click and Wait for Download ---
        
        # Get list of files before clicking, to detect the new file later
        files_before_click = set(os.listdir(download_path))

        # Click the download button
        download_button.click()
        print("Download initiated...")

        # Wait for the download to complete
        # We check the directory for a new file that isn't a temporary '.crdownload' file
        download_timeout = 30  # seconds
        start_time = time.time()
        new_file_path = None
        
        while time.time() - start_time < download_timeout:
            files_after_click = set(os.listdir(download_path))
            new_files = files_after_click - files_before_click

            if new_files:
                # Check if the new file is a temporary download file
                downloaded_file = new_files.pop()
                if not downloaded_file.endswith(('.crdownload', '.tmp')):
                    new_file_path = os.path.join(download_path, downloaded_file)
                    print(f"Success! File downloaded to: {new_file_path}")
                    break
            time.sleep(1) # Wait 1 second before checking again

        if not new_file_path:
            print("Download failed or timed out.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # --- 5. Clean Up ---
        print("Closing the browser.")
        driver.quit()

# --- Run the function ---
if __name__ == "__main__":
    download_pre_open_market_data()
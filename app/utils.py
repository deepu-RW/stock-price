import os
import time
from uuid import uuid4
from typing import Dict, Any
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from enums import TradingSignal

from fastapi import HTTPException
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import time
import pandas as pd
import pytz
import requests
from loguru import logger
# Selenium imports for auto-download
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


_instruments_cache = None
DOWNLOAD_BASE_PATH = Path("downloads")

# Ensure the directory exists, creating it along with any missing parents
def download_nse_data(session_id: str) -> Dict[str, Any]:
    """
    Downloads NSE pre-open market data using Selenium.
    Returns a dictionary with download status and file path.
    """
    # Use base download directory (no session-specific folders)
    download_path = DOWNLOAD_BASE_PATH
    
    # Create download directory if it doesn't exist
    download_path.mkdir(exist_ok=True)

    # Clean up existing CSV files BEFORE starting the download
    existing_csv_files = []
    try:
        all_files = os.listdir(download_path)
        existing_csv_files = [f for f in all_files if f.endswith('.csv')]
        
        if existing_csv_files:
            print(f"[{session_id}] Found existing CSV files: {existing_csv_files}")
            for old_csv in existing_csv_files:
                old_file_path = download_path / old_csv
                try:
                    old_file_path.unlink()  # Delete the old file
                    print(f"[{session_id}] Removed old CSV file: {old_csv}")
                except Exception as e:
                    print(f"[{session_id}] Error removing old CSV file {old_csv}: {e}")
    except Exception as e:
        print(f"[{session_id}] Error during cleanup: {e}")

    url = "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market#"
    
    # Configure Chrome options
    chrome_options = Options()
    
    # Set preferences for downloading files
    prefs = {
        "download.default_directory": str(download_path.absolute()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Essential options for server environments
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36")
    
    # Initialize WebDriver
    service = ChromeService()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, 20)
    
    try:
        print(f"[{session_id}] Navigating to: {url}")
        driver.get(url)
        
        # Wait for the download button
        print(f"[{session_id}] Waiting for download button...")
        download_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.downloads.downloadLink"))
        )
        print(f"[{session_id}] Download button found!")
        
        # Get current files before download (after cleanup)
        files_before = set(os.listdir(download_path))
        
        # Click download button
        download_button.click()
        print(f"[{session_id}] Download initiated...")
        
        # Wait for download completion
        download_timeout = 30
        start_time = time.time()
        downloaded_file_path = None
        
        while time.time() - start_time < download_timeout:
            try:
                files_after = set(os.listdir(download_path))
                new_files = files_after - files_before
                
                if new_files:
                    for file_name in new_files:
                        if not file_name.endswith(('.crdownload', '.tmp', '.part')):
                            file_path = download_path / file_name
                            # Check if file is actually complete (not 0 bytes)
                            if file_path.exists() and file_path.stat().st_size > 0:
                                downloaded_file_path = file_path
                                print(f"[{session_id}] Success! File downloaded: {file_name}")
                                break
                    
                    if downloaded_file_path:
                        break
                
                time.sleep(1)
            except Exception as e:
                print(f"[{session_id}] Error during download check: {e}")
                time.sleep(1)
        
        if not downloaded_file_path:
            return {
                "success": False,
                "error": "Download failed or timed out",
                "session_id": session_id
            }
        
        return {
            "success": True,
            "file_path": str(downloaded_file_path),
            "file_name": downloaded_file_path.name,
            "session_id": session_id,
            "replaced_files": existing_csv_files
        }
        
    except Exception as e:
        print(f"[{session_id}] Error occurred: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }
    
    finally:
        try:
            print(f"[{session_id}] Closing browser...")
            driver.quit()
        except Exception as e:
            print(f"[{session_id}] Error closing browser: {e}")



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


def run_selenium_download():
    """
    This function contains the blocking Selenium code.
    It runs in the background and saves the file to the DOWNLOAD_DIR.
    It does not return anything.
    """
    print("BACKGROUND TASK: Starting Selenium download process...")

    # --- Configure Chrome Options ---
    chrome_options = Options()
    prefs = {
        "download.default_directory": DOWNLOAD_DIR,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    # --- Initialize WebDriver ---
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        url = "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market"
        print(f"BACKGROUND TASK: Navigating to {url}")
        driver.get(url)

        download_link_locator = (By.ID, "downloadPreopen")
        wait = WebDriverWait(driver, 30)
        
        print("BACKGROUND TASK: Waiting for download link...")
        download_link_element = wait.until(
            EC.element_to_be_clickable(download_link_locator)
        )
        
        print("BACKGROUND TASK: Clicking download link.")
        download_link_element.click()

        # Wait for download to complete (up to 20 seconds)
        time.sleep(15) # Giving ample time for the download to finish
        print("BACKGROUND TASK: Download process finished.")

    except Exception as e:
        print(f"BACKGROUND TASK ERROR: An error occurred during Selenium execution: {e}")
    finally:
        print("BACKGROUND TASK: Closing the browser.")
        driver.quit()


def load_nifty_instruments():
    """Load the filtered NIFTY instruments from NIFTY.json"""
     # Read CSV, handle messy headers
    try: 
        current_date = datetime.now().strftime("%d-%b-%Y")
        file_path = f"downloads/MW-Pre-Open-Market-{current_date}.csv"
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found, call the /download endpoint to download the latest CSV file.")
        
        df = pd.read_csv(file_path) 

        # Find the correct SYMBOL column (strip and clean headers)
        cleaned_columns = [col.strip() for col in df.columns]
        df.columns = cleaned_columns

        # Sometimes the header might have trailing whitespace or newline
        symbol_col = next((col for col in df.columns if col.strip().upper() == "SYMBOL"), None)

        if symbol_col is None:
            raise ValueError("SYMBOL column not found in the CSV file")

        # Return list of symbols
        return df[symbol_col].dropna().astype(str).str.strip().tolist()
    except Exception as e:
        logger.error(f"Could not find CSV file, Call the /download endpoint to download the latest CSV file: {str(e)}")
        instruments = load_nifty_instruments()
        symbols = [inst.get("trading_symbol") for inst in instruments if inst.get("trading_symbol")]
        return symbols

def read_csv_symbols(csv_file_path: str = "data.csv", num_symbols: int = 5):
    """Read CSV file and extract specified number of symbols"""
    try:
        df = pd.read_csv(csv_file_path)
        df.columns = df.columns.str.strip()
        
        if 'SYMBOL' in df.columns:
            symbols = df['SYMBOL'].head(num_symbols).tolist()
            symbols = [str(symbol).strip() for symbol in symbols if pd.notna(symbol)]
            return symbols
        else:
            raise ValueError(f"SYMBOL column not found in CSV. Available columns: {df.columns.tolist()}")
            
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"CSV file '{csv_file_path}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")

def convert_interval_to_api_params(interval: str):
    """Convert interval string to API parameters"""
    interval_mapping = {
        "1min": ("minutes", 1),
        "15min": ("minutes", 15),
        "1hr": ("hours", 1)
    }
    
    if interval not in interval_mapping:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid interval '{interval}'. Supported intervals: {list(interval_mapping.keys())}"
        )
    
    return interval_mapping[interval]

def convert_timestamp_to_ist(timestamp):
    """Convert timestamp to IST 24-hour format"""
    try:
        if isinstance(timestamp, datetime):
            dt = timestamp
        elif isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                try:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except:
                    dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        else:
            return timestamp
        
        ist = pytz.timezone('Asia/Kolkata')
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        
        ist_dt = dt.astimezone(ist)
        return ist_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    except Exception as e:
        return timestamp

def filter_candles_by_interval(candles, interval: str, limit: int):
    """Filter candles based on interval and return latest values"""
    if not candles:
        return []
    
    processed_candles = []
    for candle in candles:
        processed_candle = candle.copy()
        if len(processed_candle) > 0:
            processed_candle[0] = convert_timestamp_to_ist(processed_candle[0])
        processed_candles.append(processed_candle)
    
    try:
        processed_candles.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S'), reverse=True)
    except:
        processed_candles.reverse()
    
    return processed_candles[:limit]

def get_historical_data(instrument_key: str, symbol: str, unit: str = "minutes", interval: int = 1, min_candles: int = 50):
    """Fetch historical candle data, going back to previous dates if needed"""
    
    payload = {}
    headers = {
        'Accept': 'application/json'
    }
    
    ist = pytz.timezone('Asia/Kolkata')
    current_date = datetime.now(ist)
    
    all_candles = []
    days_back = 0
    max_days_back = 7

    fetch_date = current_date - timedelta(days=days_back)
    to_date = fetch_date.strftime('%Y-%m-%d')
    
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
    params = {
        'to_date': to_date
    }
    
    logger.info(f"Fetching data for {symbol} - Date: {to_date}")
    
    try:
        response = requests.get(url, headers=headers, data=payload, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json().get("data", {})
            candles = data.get('candles', [])
            
            if candles:
                all_candles += candles
                logger.info(f"Fetched {len(candles)} candles for {symbol}")
        else:
            logger.error(f"API error for {to_date}: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {to_date}: {str(e)}")
    
    return all_candles

# Enhanced Technical Analysis Functions
def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return []
    
    ema = []
    multiplier = 2 / (period + 1)
    
    sma = sum(prices[:period]) / period
    ema.append(sma)
    
    for i in range(period, len(prices)):
        ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_value)
    
    return ema

def calculate_sma(prices: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return []
    
    sma = []
    for i in range(period - 1, len(prices)):
        sma.append(sum(prices[i - period + 1:i + 1]) / period)
    
    return sma

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return []
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_values = []
    
    for i in range(period, len(deltas)):
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
        
        current_gain = gains[i] if gains[i] > 0 else 0
        current_loss = losses[i] if losses[i] > 0 else 0
        
        avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
        avg_loss = ((avg_loss * (period - 1)) + current_loss) / period
    
    return rsi_values

def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow_period:
        return {"macd": [], "signal": [], "histogram": []}
    
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    # Align EMAs to same length
    min_len = min(len(ema_fast), len(ema_slow))
    ema_fast = ema_fast[-min_len:]
    ema_slow = ema_slow[-min_len:]
    
    macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
    signal_line = calculate_ema(macd_line, signal_period)
    
    # Calculate histogram
    histogram = []
    if len(signal_line) > 0:
        signal_start = len(macd_line) - len(signal_line)
        for i in range(len(signal_line)):
            histogram.append(macd_line[signal_start + i] - signal_line[i])
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }

def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict:
    """Calculate Bollinger Bands"""[1][4][13]
    if len(prices) < period:
        return {"upper": [], "middle": [], "lower": []}
    
    sma = calculate_sma(prices, period)
    upper_band = []
    lower_band = []
    
    for i in range(period - 1, len(prices)):
        period_prices = prices[i - period + 1:i + 1]
        std = (sum([(p - sma[i - period + 1]) ** 2 for p in period_prices]) / period) ** 0.5
        
        upper_band.append(sma[i - period + 1] + (std_dev * std))
        lower_band.append(sma[i - period + 1] - (std_dev * std))
    
    return {
        "upper": upper_band,
        "middle": sma,
        "lower": lower_band
    }

def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict:
    """Calculate Average Directional Index (ADX)"""[21][24][27]
    if len(highs) < period + 1:
        return {"adx": [], "di_plus": [], "di_minus": []}
    
    # Calculate True Range
    tr_values = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close_prev = abs(highs[i] - closes[i-1])
        low_close_prev = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_close_prev, low_close_prev)
        tr_values.append(tr)
    
    # Calculate Directional Movement
    dm_plus = []
    dm_minus = []
    for i in range(1, len(highs)):
        move_up = highs[i] - highs[i-1]
        move_down = lows[i-1] - lows[i]
        
        if move_up > move_down and move_up > 0:
            dm_plus.append(move_up)
        else:
            dm_plus.append(0)
            
        if move_down > move_up and move_down > 0:
            dm_minus.append(move_down)
        else:
            dm_minus.append(0)
    
    # Calculate smoothed averages
    atr = calculate_sma(tr_values, period)
    di_plus_smooth = calculate_sma(dm_plus, period)
    di_minus_smooth = calculate_sma(dm_minus, period)
    
    # Calculate DI+ and DI-
    di_plus = [(dm / atr[i]) * 100 for i, dm in enumerate(di_plus_smooth) if i < len(atr)]
    di_minus = [(dm / atr[i]) * 100 for i, dm in enumerate(di_minus_smooth) if i < len(atr)]
    
    # Calculate ADX
    dx_values = []
    for i in range(len(di_plus)):
        if di_plus[i] + di_minus[i] != 0:
            dx = abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i]) * 100
            dx_values.append(dx)
    
    adx_values = calculate_sma(dx_values, period)
    
    return {
        "adx": adx_values,
        "di_plus": di_plus,
        "di_minus": di_minus
    }

def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Dict:
    """Calculate Stochastic Oscillator"""[42][45][48]
    if len(highs) < k_period:
        return {"k_percent": [], "d_percent": []}
    
    k_percent = []
    for i in range(k_period - 1, len(closes)):
        period_high = max(highs[i - k_period + 1:i + 1])
        period_low = min(lows[i - k_period + 1:i + 1])
        
        if period_high != period_low:
            k_value = ((closes[i] - period_low) / (period_high - period_low)) * 100
        else:
            k_value = 50
        
        k_percent.append(k_value)
    
    d_percent = calculate_sma(k_percent, d_period)
    
    return {
        "k_percent": k_percent,
        "d_percent": d_percent
    }

def calculate_williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """Calculate Williams %R"""[61][64][67]
    if len(highs) < period:
        return []
    
    williams_r = []
    for i in range(period - 1, len(closes)):
        period_high = max(highs[i - period + 1:i + 1])
        period_low = min(lows[i - period + 1:i + 1])
        
        if period_high != period_low:
            wr_value = ((period_high - closes[i]) / (period_high - period_low)) * -100
        else:
            wr_value = -50
        
        williams_r.append(wr_value)
    
    return williams_r

def calculate_cci(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> List[float]:
    """Calculate Commodity Channel Index"""[62][65][68]
    if len(highs) < period:
        return []
    
    typical_prices = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(highs))]
    sma_tp = calculate_sma(typical_prices, period)
    
    cci_values = []
    for i in range(period - 1, len(typical_prices)):
        period_tp = typical_prices[i - period + 1:i + 1]
        mean_deviation = sum([abs(tp - sma_tp[i - period + 1]) for tp in period_tp]) / period
        
        if mean_deviation != 0:
            cci = (typical_prices[i] - sma_tp[i - period + 1]) / (0.015 * mean_deviation)
        else:
            cci = 0
        
        cci_values.append(cci)
    
    return cci_values

def calculate_obv(closes: List[float], volumes: List[float]) -> List[float]:
    """Calculate On-Balance Volume"""[63][66][69]
    if len(closes) != len(volumes) or len(closes) < 2:
        return []
    
    obv = [volumes[0]]
    
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    
    return obv

def calculate_supertrend(highs: List[float], lows: List[float], closes: List[float], period: int = 10, multiplier: float = 3.0) -> Dict:
    """Calculate SuperTrend Indicator"""[41][44][47]
    if len(highs) < period:
        return {"supertrend": [], "trend": []}
    
    # Calculate ATR
    tr_values = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close_prev = abs(highs[i] - closes[i-1])
        low_close_prev = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_close_prev, low_close_prev)
        tr_values.append(tr)
    
    atr = calculate_sma(tr_values, period)
    
    hl2 = [(highs[i] + lows[i]) / 2 for i in range(len(highs))]
    
    supertrend = []
    trend = []
    
    for i in range(period, len(closes)):
        atr_index = i - period
        if atr_index < len(atr):
            basic_upper = hl2[i] + (multiplier * atr[atr_index])
            basic_lower = hl2[i] - (multiplier * atr[atr_index])
            
            # Determine trend direction
            if len(supertrend) == 0:
                supertrend.append(basic_lower)
                trend.append(1)  # 1 for uptrend
            else:
                if closes[i] <= supertrend[-1]:
                    supertrend.append(basic_upper)
                    trend.append(-1)  # -1 for downtrend
                else:
                    supertrend.append(basic_lower)
                    trend.append(1)
    
    return {
        "supertrend": supertrend,
        "trend": trend
    }

def extract_prices_and_volumes(candles: List[List]) -> Tuple[List[float], List[float]]:
    """Extract close prices and volumes from candle data with enhanced validation"""
    prices, volumes = [], []
    
    if not candles:
        logger.warning("Empty candles data received")
        return prices, volumes
    
    for i, candle in enumerate(candles):
        try:
            # Validate candle structure
            if not isinstance(candle, (list, tuple)):
                logger.warning(f"Candle {i} is not a list/tuple: {type(candle)}")
                continue
                
            if len(candle) < 6:
                logger.warning(f"Candle {i} has insufficient data points: {len(candle)} (expected at least 6)")
                continue
            
            # Safely extract data with type conversion
            close_price = float(candle[4]) if candle[4] is not None else 0.0
            volume = float(candle[5]) if candle[5] is not None else 0.0
            
            prices.append(close_price)
            volumes.append(volume)
            
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error processing candle {i}: {candle} - {str(e)}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing candle {i}: {str(e)}")
            continue
    
    return prices, volumes


def extract_hlcv(candles: List[List]) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Extract high, low, close, and volume from candle data with enhanced validation"""
    high, low, close, volume = [], [], [], []
    
    if not candles:
        logger.warning("Empty candles data received")
        return high, low, close, volume
    
    for i, candle in enumerate(candles):
        try:
            # Validate candle structure
            if not isinstance(candle, (list, tuple)):
                logger.warning(f"Candle {i} is not a list/tuple: {type(candle)}")
                continue
                
            if len(candle) < 6:
                logger.warning(f"Candle {i} has insufficient data points: {len(candle)} (expected at least 6)")
                continue
            
            # Safely extract data with type conversion
            high_val = float(candle[2]) if candle[2] is not None else 0.0
            low_val = float(candle[3]) if candle[3] is not None else 0.0
            close_val = float(candle[4]) if candle[4] is not None else 0.0
            volume_val = float(candle[5]) if candle[5] is not None else 0.0
            
            high.append(high_val)
            low.append(low_val)
            close.append(close_val)
            volume.append(volume_val)
            
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error processing candle {i}: {candle} - {str(e)}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing candle {i}: {str(e)}")
            continue
    
    return high, low, close, volume


def calculate_vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> float:
    """Calculate Volume Weighted Average Price"""
    if not (len(highs) == len(lows) == len(closes) == len(volumes)):
        raise ValueError("All input lists must have the same length")
    
    cumulative_pv = 0
    cumulative_volume = 0
    
    for i in range(len(highs)):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3
        cumulative_pv += typical_price * volumes[i]
        cumulative_volume += volumes[i]
    
    if cumulative_volume == 0:
        return 0
    
    return cumulative_pv / cumulative_volume

def calculate_confidence_score(conditions_met: Dict[str, bool], weights: Dict[str, float]) -> float:
    """Calculate confidence score based on weighted conditions"""
    total_weight = sum(weights.values())
    achieved_weight = sum(weights[condition] for condition, met in conditions_met.items() if met)
    
    if total_weight == 0:
        return 0.0
    
    confidence = (achieved_weight / total_weight) * 100
    return min(confidence, 100.0)

def analyze_trading_signal(candles_1m: List[List], candles_5m: List[List], candles_15m: List[List], current_price: float) -> TradingSignal:
    """Enhanced trading signal analysis with comprehensive error handling"""
    logger.info("Analyzing trading signal with enhanced error handling")
    
    reasons = []
    technical_indicators = {}
    signal = "HOLD"
    
    
    try:
        # Validate input data first
        if not candles_1m or not candles_5m:
            return TradingSignal(
                signal="HOLD", confidence=0.0,
                reasons=["Insufficient candle data provided"],
                technical_indicators={}
            )
        
        # Log data structure for debugging
        logger.info(f"1m candles count: {len(candles_1m)}")
        logger.info(f"5m candles count: {len(candles_5m)}")
        
        if len(candles_1m) > 0:
            logger.info(f"Sample 1m candle structure: {candles_1m[0] if candles_1m[0] else 'Empty'}")
        if len(candles_5m) > 0:
            logger.info(f"Sample 5m candle structure: {candles_5m[0] if candles_5m[0] else 'Empty'}")
        
        # Extract data with enhanced validation
        prices_1m, volumes_1m = extract_prices_and_volumes(candles_1m)
        prices_5m, volumes_5m = extract_prices_and_volumes(candles_5m)
        
        # Check if we have enough data after extraction
        if len(prices_1m) < 30 or len(prices_5m) < 15:
            return TradingSignal(
                signal="HOLD", confidence=0.0,
                reasons=[f"Insufficient valid data after processing: 1m={len(prices_1m)}, 5m={len(prices_5m)}"],
                technical_indicators={}
            )
        
        # Extract HLCV data with validation
        high_1m, low_1m, close_1m, volume_1m = extract_hlcv(candles_1m)
        
        if len(high_1m) != len(low_1m) or len(low_1m) != len(close_1m):
            return TradingSignal(
                signal="HOLD", confidence=0.0,
                reasons=["HLCV data arrays have mismatched lengths"],
                technical_indicators={}
            )
        
         # Calculate EMAs - Focus on most common periods
        ema9_1m = calculate_ema(prices_1m, 9)
        ema21_1m = calculate_ema(prices_1m, 21)
        ema9_5m = calculate_ema(prices_5m, 9)
        ema21_5m = calculate_ema(prices_5m, 21)
        
        # Calculate RSI
        rsi_1m = calculate_rsi(prices_1m)
        
        # Calculate VWAP
        high_1m, low_1m, close_1m, volume_1m = extract_hlcv(candles_1m)
        vwap_1m = calculate_vwap(high_1m, low_1m, close_1m, volume_1m)
        
        # Volume analysis
        avg_volume_1m = sum(volumes_1m[-20:]) / len(volumes_1m[-20:]) if len(volumes_1m) >= 20 else None
        current_volume = volumes_1m[-1] if volumes_1m else 0
        
        # Store technical indicators
        technical_indicators = {
            "1m": {
                "current_price": prices_1m[-1],
                "ema9": ema9_1m[-1] if ema9_1m else None,
                "ema21": ema21_1m[-1] if ema21_1m else None,
                "rsi": rsi_1m[-1] if rsi_1m else None,
                "vwap": vwap_1m,
                "volume_ratio": current_volume / avg_volume_1m if avg_volume_1m else None
            },
            "5m": {
                "ema9": ema9_5m[-1] if ema9_5m else None,
                "ema21": ema21_5m[-1] if ema21_5m else None
            }
        }
        
        # BUY CONDITIONS
        buy_conditions = {}

        buy_weights = {
            "ema_bullish_alignment": 35.0,
            "rsi_momentum": 25.0,
            "vwap_breakout": 25.0,
            "volume_confirmation": 15.0
        }
        
        sell_weights = {
            "ema_bearish_breakdown": 35.0,
            "rsi_overbought": 25.0,
            "vwap_rejection": 25.0,
            "momentum_loss": 15.0
        }
        
        # 1. EMA Bullish Alignment (Combined EMA condition)
        if ema9_1m and ema21_1m and ema9_5m and ema21_5m:
            ema_1m_bullish = ema9_1m[-1] > ema21_1m[-1] and prices_1m[-1] > ema9_1m[-1]
            ema_5m_bullish = ema9_5m[-1] > ema21_5m[-1]
            buy_conditions["ema_bullish_alignment"] = ema_1m_bullish and ema_5m_bullish
            
            if buy_conditions["ema_bullish_alignment"]:
                reasons.append("✅ EMA Bullish Alignment: 9>21 on both 1m & 5m + price above 1m EMA9")
            else:
                reasons.append("❌ EMA alignment not bullish")
        
        # 2. RSI Momentum (30-70 range with upward momentum)
        if rsi_1m and len(rsi_1m) >= 2:
            rsi_current = rsi_1m[-1]
            rsi_previous = rsi_1m[-2]
            buy_conditions["rsi_momentum"] = 30 < rsi_current < 70 and rsi_current > rsi_previous
            
            if buy_conditions["rsi_momentum"]:
                reasons.append(f"✅ RSI Momentum: {rsi_current:.1f} (30-70 range, rising)")
            else:
                reasons.append(f"❌ RSI: {rsi_current:.1f} (not in momentum range or falling)")
        
        # 3. VWAP Breakout
        if vwap_1m:
            buy_conditions["vwap_breakout"] = current_price > vwap_1m * 1.002  # 0.2% above VWAP
            
            if buy_conditions["vwap_breakout"]:
                reasons.append(f"✅ VWAP Breakout: Price {current_price:.2f} > VWAP {vwap_1m:.2f}")
            else:
                reasons.append(f"❌ Current Price {current_price:.2f} not above VWAP {vwap_1m:.2f}")
        
        # 4. Volume Confirmation
        if avg_volume_1m:
            buy_conditions["volume_confirmation"] = current_volume > 1.5 * avg_volume_1m
            
            if buy_conditions["volume_confirmation"]:
                reasons.append(f"✅ Volume Spike: {current_volume:.0f} greater than 1.5x avg")
            else:
                reasons.append("❌ No significant volume spike")
        
        # SELL CONDITIONS
        sell_conditions = {}
        
        # 1. EMA Bearish Breakdown (Combined EMA condition)
        if ema9_1m and ema21_1m:
            price_below_ema9 = prices_1m[-1] < ema9_1m[-1]
            ema_bearish = ema9_1m[-1] < ema21_1m[-1]
            sell_conditions["ema_bearish_breakdown"] = price_below_ema9 or ema_bearish
            
            if sell_conditions["ema_bearish_breakdown"]:
                reasons.append("🔴 EMA Bearish Breakdown: Price below EMA9 or EMA9 less than EMA21")
            else:
                reasons.append("✅ EMA structure remains bullish")
        
        # 2. RSI Overbought with Bearish Divergence
        if rsi_1m and len(rsi_1m) >= 5:
            rsi_current = rsi_1m[-1]
            rsi_overbought = rsi_current > 70
            
            # Check for bearish divergence
            price_higher = prices_1m[-1] > max(prices_1m[-5:-1])
            rsi_lower = rsi_current < max(rsi_1m[-5:-1])
            bearish_divergence = price_higher and rsi_lower
            
            sell_conditions["rsi_overbought"] = rsi_overbought or bearish_divergence
            
            if sell_conditions["rsi_overbought"]:
                divergence_text = " with bearish divergence" if bearish_divergence else ""
                reasons.append(f"🔴 RSI: {rsi_current:.1f} (overbought{divergence_text})")
            else:
                reasons.append(f"✅ RSI: {rsi_current:.1f} (not overbought)")
        
        # 3. VWAP Rejection
        if vwap_1m:
            sell_conditions["vwap_rejection"] = current_price < vwap_1m * 0.998  # 0.2% below VWAP
            
            if sell_conditions["vwap_rejection"]:
                reasons.append(f"🔴 VWAP Rejection: Price {current_price:.2f} < VWAP {vwap_1m:.2f}")
            else:
                reasons.append(f"✅ Price holding above VWAP")
        
        # 4. Momentum Loss (Price fails to make higher highs)
        if len(prices_1m) >= 10:
            recent_high = max(prices_1m[-10:])
            momentum_loss = prices_1m[-1] < recent_high * 0.995  # 0.5% below recent high
            
            sell_conditions["momentum_loss"] = momentum_loss
            
            if sell_conditions["momentum_loss"]:
                reasons.append("🔴 Momentum Loss: Failed to sustain near recent highs")
            else:
                reasons.append("✅ Momentum intact")
        
        # SIGNAL DETERMINATION
        buy_confidence = calculate_confidence_score(buy_conditions, buy_weights)
        sell_confidence = calculate_confidence_score(sell_conditions, sell_weights)
        
        # More decisive thresholds
        if sell_confidence > 70:
            signal = "SELL"
            confidence = sell_confidence
        elif buy_confidence > 70:
            signal = "BUY"
            confidence = buy_confidence
        else:
            signal = "HOLD"
            confidence = max(buy_confidence, sell_confidence)
        
        reasons.append(f"📊 Final - Buy: {buy_confidence:.1f}% | Sell: {sell_confidence:.1f}%")

        return TradingSignal(
                signal=signal, confidence=confidence,
                reasons=reasons, technical_indicators=technical_indicators
            )
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {str(e)}", exc_info=True)
        return TradingSignal(
            signal="HOLD", confidence=0.0,
            reasons=[f"Enhanced analysis error: {str(e)}"],
            technical_indicators={}
        )


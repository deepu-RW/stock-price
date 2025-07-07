from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
import requests
import json
import os
import pandas as pd
import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import pytz
from enum import Enum
from loguru import logger
import sys
import numpy as np
from typing import Tuple

# Selenium imports for auto-download
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException

# Configure logging
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="ERROR")

app = FastAPI(
    title="Advanced NIFTY Trading System",
    description="Comprehensive API for NSE data download and trading analysis with automated pre-open data fetching",
    version="2.0.0"
)

# Enum for interval values
class IntervalEnum(str, Enum):
    one_min = "1min"
    fifteen_min = "15min"
    one_hour = "1hr"

# Response models
class CandleData(BaseModel):
    symbol: str
    instrument_key: str
    candles: List[List]
    metadata: Dict[str, Any]

class AvailableSymbolsResponse(BaseModel):
    total_symbols: int
    symbols: List[str]

class ErrorResponse(BaseModel):
    error: str
    message: str

class MultipleSymbolsResponse(BaseModel):
    processed_symbols: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, str]]

class TradingSignal(BaseModel):
    signal: str  # "BUY", "HOLD", "SELL"
    confidence: float  # 0-100
    reasons: List[str]
    technical_indicators: Dict[str, Any]

class TradingStatusResponse(BaseModel):
    symbol: str
    current_price: float
    signal: str
    confidence: float
    reasons: List[str]
    technical_analysis: Dict[str, Any]
    timestamp: str

class DownloadStatusResponse(BaseModel):
    status: str
    message: str
    download_path: str
    timestamp: str
    file_size: Optional[int] = None

# Global variable to cache instruments
_instruments_cache = None

def setup_selenium_driver():
    """Setup Chrome driver with download preferences"""
    download_path = os.getcwd()
    
    chrome_options = Options()
    prefs = {
        "download.default_directory": download_path,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Run in headless mode for production
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        logger.error(f"Failed to setup Chrome driver: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Driver setup failed: {str(e)}")

def download_nse_preopen_data():
    """Download NSE pre-open data using Selenium"""
    driver = None
    try:
        logger.info("Starting NSE pre-open data download...")
        driver = setup_selenium_driver()
        
        url = "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market"
        logger.info(f"Navigating to: {url}")
        driver.get(url)

        # Wait for the download link to be clickable
        download_link_locator = (By.ID, "downloadPreopen")
        wait = WebDriverWait(driver, 30)
        download_link_element = wait.until(
            EC.element_to_be_clickable(download_link_locator)
        )

        logger.info("Download link found. Clicking to start download...")
        download_link_element.click()

        # Wait for download to complete
        time.sleep(15)  # Increased wait time for large files
        
        logger.info("NSE pre-open data download completed successfully")
        return True
        
    except TimeoutException:
        logger.error("Timeout waiting for download link")
        return False
    except WebDriverException as e:
        logger.error(f"WebDriver error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {str(e)}")
        return False
    finally:
        if driver:
            driver.quit()

def load_nifty_instruments():
    """Load the filtered NIFTY instruments from NIFTY.json"""
    global _instruments_cache
    
    if _instruments_cache is not None:
        return _instruments_cache
    try:
        with open("NIFTY.json", "r") as file:
            instruments = json.load(file)
            _instruments_cache = instruments
            return instruments
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading NIFTY.json: {str(e)}")

def get_instrument_key(instruments, symbol):
    """Find instrument key for a given trading symbol"""
    for instrument in instruments:
        if instrument.get("trading_symbol") == symbol:
            return instrument.get("instrument_key")
    return None

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

# Technical Analysis Functions
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

def calculate_rsi(prices: List[float]) -> List[float]:
    """Calculate Relative Strength Index"""
    period = 14
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

def extract_prices_and_volumes(candles: List[List]) -> Tuple[List[float], List[float]]:
    """Extract close prices and volumes from candle data"""
    if not candles or len(candles[0]) < 6:
        return [], []
    
    prices = [float(candle[4]) for candle in candles]
    volumes = [float(candle[5]) for candle in candles]
    
    return prices, volumes

def extract_hlcv(candles: List[List]) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Extract high, low, close, and volume from candle data"""
    high, low, close, volume = [], [], [], []
    for candle in candles:
        if len(candle) > 5:
            high.append(candle[2])
            low.append(candle[3])
            close.append(candle[4])
            volume.append(candle[5])
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
    """Analyze trading signal with focused conditions for intraday trading"""
    logger.info("Analyzing trading signal with new buy/sell conditions")
    
    reasons = []
    technical_indicators = {}
    signal = "HOLD"
    
    # Simplified weights - fewer conditions
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
    
    try:
        prices_1m, volumes_1m = extract_prices_and_volumes(candles_1m)
        prices_5m, volumes_5m = extract_prices_and_volumes(candles_5m)
        
        if len(prices_1m) < 30 or len(prices_5m) < 15:
            return TradingSignal(
                signal="HOLD", confidence=0.0,
                reasons=["Insufficient data for analysis"],
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
                reasons.append(f"❌ Price {current_price:.2f} not above VWAP {vwap_1m:.2f}")
        
        # 4. Volume Confirmation
        if avg_volume_1m:
            buy_conditions["volume_confirmation"] = current_volume > 1.5 * avg_volume_1m
            
            if buy_conditions["volume_confirmation"]:
                reasons.append(f"✅ Volume Spike: {current_volume:.0f} > 1.5x avg")
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
                reasons.append("🔴 EMA Bearish Breakdown: Price below EMA9 or EMA9 < EMA21")
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
        
    except Exception as e:
        return TradingSignal(
            signal="HOLD", confidence=0.0,
            reasons=[f"Analysis error: {str(e)}"],
            technical_indicators={}
        )
    
    return TradingSignal(
        signal=signal, confidence=confidence,
        reasons=reasons, technical_indicators=technical_indicators
    )
# API ENDPOINTS

@app.get("/", summary="API Information")
async def root():
    """Root endpoint with comprehensive API information"""
    return {
        "message": "Advanced NIFTY Trading System",
        "version": "2.0.0",
        "features": [
            "Automated NSE pre-open data download",
            "Multi-timeframe technical analysis",
            "Advanced trading signals",
            "Real-time market data processing"
        ],
        "endpoints": {
            "/download-preopen": "Download latest NSE pre-open data",
            "/symbols": "Get all available NIFTY symbols",
            "/historical-data/{symbol}/{limit}/{interval}": "Get historical data",
            "/{symbol}/decide": "Get trading recommendation",
            "/process-csv-symbols": "Process symbols from CSV",
            "/health": "System health check"
        },
        "supported_intervals": ["1min", "15min", "1hr"],
        "timestamp_format": "IST 24-hour format (YYYY-MM-DD HH:MM:SS)"
    }

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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


# --- 3. THE SELENIUM LOGIC (DESIGNED FOR BACKGROUND EXECUTION) ---

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

# --- 4. DEFINE THE API ENDPOINTS --

@app.get(
    "/download-preopen", 
    response_model=DownloadStatusResponse, 
    summary="Initiate NSE Pre-Open Data Download",
    tags=["Data Downloader"]
)
async def initiate_download_endpoint(background_tasks: BackgroundTasks):
    """
    Initiates the download of the pre-open market data in the background.
    This endpoint returns immediately with a confirmation message.
    """
    print("API ENDPOINT: Received request to download pre-open data.")
    
    # Add the long-running Selenium function as a background task
    background_tasks.add_task(run_selenium_download)
    
    # Return an immediate response to the user
    return DownloadStatusResponse(
        status="initiated",
        message="Download of pre-open market data has been initiated. "
                "Check the status or retrieve the file in a few moments."
    )

@app.get("/symbols", response_model=AvailableSymbolsResponse, summary="Get Available Symbols")
async def get_available_symbols():
    """Get all available NIFTY symbols from the JSON file"""
    instruments = load_nifty_instruments()
    
    symbols = [inst.get("trading_symbol") for inst in instruments if inst.get("trading_symbol")]
    symbols.sort()
    
    return AvailableSymbolsResponse(
        total_symbols=len(symbols),
        symbols=symbols
    )

@app.get("/historical-data/{symbol}/{limit}/{interval}", response_model=CandleData, summary="Get Historical Data")
async def get_historical_data_with_interval(
    symbol: str,
    limit: int,
    interval: IntervalEnum
):
    """Get historical candle data for a specific NIFTY symbol with specified interval"""
    
    symbol = symbol.upper()
    instruments = load_nifty_instruments()
    instrument_key = get_instrument_key(instruments, symbol)
    
    if not instrument_key:
        raise HTTPException(
            status_code=404, 
            detail=f"Symbol '{symbol}' not found in NIFTY instruments."
        )
    
    unit, api_interval = convert_interval_to_api_params(interval)
    candles = get_historical_data(instrument_key, symbol, unit, api_interval)
    processed_candles = filter_candles_by_interval(candles, interval, limit)
    
    ist = pytz.timezone('Asia/Kolkata')
    current_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
    
    return CandleData(
        symbol=symbol,
        instrument_key=instrument_key,
        candles=processed_candles,
        metadata={
            "total_candles": len(processed_candles),
            "requested_limit": limit,
            "interval": interval,
            "timezone": "IST (Asia/Kolkata)",
            "current_ist_time": current_ist
        }
    )

# Updated endpoint call
@app.get("/{symbol}/decide", response_model=TradingStatusResponse, summary="Get Trading Status")
async def get_trading_status(symbol: str):
    """
    Get trading recommendation (BUY/SELL/HOLD) for a symbol based on focused multi-timeframe analysis
    
    Focused Strategy:
    - Primary: EMA(9) vs EMA(21) on 1m, EMA(5) vs EMA(13) on 5m
    - RSI momentum and overbought/oversold conditions
    - Volume analysis for confirmation
    - MVWAP positioning for price context
    - Comprehensive sell signals for exit strategies
    """
    print("trug")
    symbol = symbol.upper()
    logger.info(f"Analyzing trading status for {symbol}")
    
    # Load instruments and get instrument key
    instruments = load_nifty_instruments()
    instrument_key = get_instrument_key(instruments, symbol)
    
    if not instrument_key:
        raise HTTPException(
            status_code=404, 
            detail=f"Symbol '{symbol}' not found in NIFTY instruments."
        )
    
    try:
        # Fetch multi-timeframe data with reduced periods for faster response
        candles_1m = get_historical_data(instrument_key, symbol, "minutes", 1, 50)
        candles_5m = get_historical_data(instrument_key, symbol, "minutes", 5, 25)
        candles_15m = get_historical_data(instrument_key, symbol, "minutes", 15, 15)
        
        current_price = float(candles_1m[0][4]) if candles_1m else 0.0
        
        # Analyze with improved signal logic
        trading_signal = analyze_trading_signal(candles_1m, candles_5m, candles_15m, current_price)
        
        # Get current IST time
        ist = pytz.timezone('Asia/Kolkata')
        current_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
        
        return TradingStatusResponse(
            symbol=symbol,
            current_price=current_price,
            signal=trading_signal.signal,
            confidence=trading_signal.confidence,
            reasons=trading_signal.reasons,
            technical_analysis=trading_signal.technical_indicators,
            timestamp=current_ist
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing trading status: {str(e)}"
        )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import uuid
from pathlib import Path

# app = FastAPI(title="NSE Pre-Open Market Data Downloader", version="1.0.0")

DOWNLOAD_BASE_PATH = Path("downloads")
if not DOWNLOAD_BASE_PATH.exists():
    DOWNLOAD_BASE_PATH.mkdir(parents=True, exist_ok=True)

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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "NSE Pre-Open Market Data Downloader API", "status": "active"}

@app.get("/download-nse-data")
async def download_nse_data_endpoint():
    """
    Downloads NSE pre-open market data and returns download information.
    """
    session_id = str(uuid.uuid4())
    
    try:
        # Run the download in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, download_nse_data, session_id)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        response_data = {
            "message": "Download completed successfully",
            "session_id": session_id,
            "file_name": result["file_name"],
            "download_url": f"/download-file/{result['file_name']}"
        }
        
        # Add information about replaced files if any
        if result.get("replaced_files"):
            response_data["replaced_files"] = result["replaced_files"]
            response_data["message"] += f" (replaced {len(result['replaced_files'])} existing CSV file(s))"
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
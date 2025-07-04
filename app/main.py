from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import requests
import json
import os
import pandas as pd
import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import pytz
from enum import Enum
from datetime import datetime, timedelta
from loguru import logger
import sys

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="ERROR")

app = FastAPI(
    title="NIFTY Historical Data API",
    description="API to fetch historical candle data for NIFTY instruments",
    version="1.1.0"
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

# Global variable to cache instruments
_instruments_cache = None

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
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Clean column names (remove trailing spaces and newlines)
        df.columns = df.columns.str.strip()
        
        # Extract symbols from the SYMBOL column
        if 'SYMBOL' in df.columns:
            symbols = df['SYMBOL'].head(num_symbols).tolist()
            # Clean symbols (remove any trailing spaces)
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
        # If timestamp is already a datetime object
        if isinstance(timestamp, datetime):
            dt = timestamp
        # If timestamp is a string, parse it
        elif isinstance(timestamp, str):
            # Try different timestamp formats
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                try:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                except:
                    dt = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
        # If timestamp is a number (Unix timestamp)
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        else:
            return timestamp  # Return as-is if we can't parse it
        
        # Convert to IST
        ist = pytz.timezone('Asia/Kolkata')
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        
        ist_dt = dt.astimezone(ist)
        return ist_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    except Exception as e:
        # If conversion fails, return original timestamp
        return timestamp

def filter_candles_by_interval(candles, interval: str, limit: int):
    """Filter candles based on interval and return latest values"""
    if not candles:
        return []
    
    # Convert timestamps to IST and sort by timestamp (most recent first)
    processed_candles = []
    for candle in candles:
        processed_candle = candle.copy()
        if len(processed_candle) > 0:
            # Assuming first element is timestamp
            processed_candle[0] = convert_timestamp_to_ist(processed_candle[0])
        processed_candles.append(processed_candle)
    print("processed_candles: ", processed_candles)
    
    # Sort by timestamp (most recent first)
    try:
        processed_candles.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S'), reverse=True)
    except:
        # If sorting fails, just reverse the list
        processed_candles.reverse()
    
    # Return the requested number of latest candles
    return processed_candles[:limit]


def get_historical_data(instrument_key: str, symbol: str, unit: str = "minutes", interval: int = 1, min_candles: int = 50):
    """Fetch historical candle data, going back to previous dates if needed"""
    
    payload = {}
    headers = {
        'Accept': 'application/json'
    }
    
    # Get current date in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_date = datetime.now(ist)
    
    all_candles = []
    days_back = 0
    max_days_back = 7  # Don't go back more than 7 days

    fetch_date = current_date - timedelta(days=days_back)
    to_date = fetch_date.strftime('%Y-%m-%d')
    print("Fetch date: ", fetch_date)
    print("To date: ", to_date)
    print("Instrument key:", instrument_key)
    print("Symbol:", symbol)
    print("Unit:", unit)
    print("Interval:", interval)
    print("To date:", to_date)
    
    # Build the URL with date parameters
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
    params = {
        'to_date': to_date
    }
    
    print(f"Fetching data for {symbol} - Date: {to_date}, URL: {url}")
    
    try:
        response = requests.get(url, headers=headers, data=payload, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json().get("data", {})
            print("DATA: ", data)
            candles = data.get('candles', [])
            
            if candles:
                # Add candles to the beginning of the list (older candles first)
                all_candles += candles
                print(f"Fetched {len(candles)} candles with {interval} {unit} interval for {to_date}. Total: {len(all_candles)}")
            
            days_back += 1
        else:
            print(f"API error for {to_date}: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request error for {to_date}: {str(e)}")
        days_back += 1

    print("ALL CANDLES: ", all_candles)
    # if len(all_candles) < min_candles:
    #     raise HTTPException(
    #         status_code=503, 
    #         detail=f"Could not fetch enough candles. Got {len(all_candles)}, need {min_candles}"
    #     )
    
    return all_candles

# Update the original function to use the new one
async def fetch_historical_data_async(session, base_url: str, symbol: str, limit: Optional[int] = None, unit: str = "minutes", interval: int = 1):
    """Async function to fetch historical data for a symbol"""
    try:
        if limit is not None:
            url = f"{base_url}/historical-data/{symbol}/{limit}?unit={unit}&interval={interval}"
        else:
            url = f"{base_url}/historical-data/{symbol}?unit={unit}&interval={interval}"
            
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "symbol": symbol,
                    "status": "success",
                    "data": data
                }
            else:
                error_text = await response.text()
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": f"HTTP {response.status}: {error_text}"
                }
    except Exception as e:
        return {
            "symbol": symbol,
            "status": "error",
            "error": str(e)
        }

@app.get("/", summary="API Information")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NIFTY Historical Data API",
        "version": "1.1.0",
        "endpoints": {
            "/symbols": "Get all available NIFTY symbols",
            "/historical-data/{symbol}/{limit}/{interval}": "Get historical data with specific interval (1min, 15min, 1hr)",
            "/process-csv-symbols": "Process symbols from CSV and fetch historical data",
            "/csv-symbols": "Get symbols from CSV file",
            "/docs": "Interactive API documentation"
        },
        "supported_intervals": ["1min", "15min", "1hr"],
        "timestamp_format": "IST 24-hour format (YYYY-MM-DD HH:MM:SS)"
    }

@app.get("/csv-symbols", summary="Get Symbols from CSV")
async def get_csv_symbols(num_symbols: int = Query(default=5, description="Number of symbols to extract", ge=1, le=50)):
    """Extract symbols from the CSV file"""
    symbols = read_csv_symbols(num_symbols=num_symbols)
    return {
        "total_symbols_extracted": len(symbols),
        "symbols": symbols,
        "source": "data.csv"
    }

@app.get("/process-csv-symbols", response_model=MultipleSymbolsResponse, summary="Process CSV Symbols")
async def process_csv_symbols(
    num_symbols: int = Query(default=5, description="Number of symbols to process", ge=1, le=20),
    limit: Optional[int] = Query(default=None, description="Number of most recent candlesticks to return per symbol"),
    unit: str = Query(default="minutes", description="Time unit (minutes, hours, days)"),
    interval: int = Query(default=1, description="Interval value", ge=1),
    use_internal_api: bool = Query(default=True, description="Use internal API endpoints vs external calls")
):
    """
    Process symbols from CSV file and fetch historical data for each
    
    - **num_symbols**: Number of symbols to process from CSV (default: 5)
    - **limit**: Number of most recent candlesticks to return per symbol (optional)
    - **unit**: Time unit - minutes, hours, or days (default: minutes)
    - **interval**: Interval value (default: 1)
    - **use_internal_api**: If True, uses internal functions; if False, makes HTTP calls to own endpoints
    """
    
    # Get symbols from CSV
    symbols = read_csv_symbols(num_symbols=num_symbols)
    
    results = []
    errors = []
    
    if use_internal_api:
        # Use internal functions directly
        instruments = load_nifty_instruments()
        
        for symbol in symbols:
            try:
                symbol = symbol.upper()
                instrument_key = get_instrument_key(instruments, symbol)
                
                if not instrument_key:
                    errors.append({
                        "symbol": symbol,
                        "error": f"Symbol '{symbol}' not found in NIFTY instruments"
                    })
                    continue
                
                candles = get_historical_data(instrument_key, symbol, unit, interval)
                
                # Apply limit if specified (get most recent candles)
                original_count = len(candles)
                if limit is not None and limit > 0:
                    candles = candles[-limit:] if len(candles) > limit else candles
                
                results.append({
                    "symbol": symbol,
                    "instrument_key": instrument_key,
                    "candles": candles,
                    "metadata": {
                        "total_candles": len(candles),
                        "original_total_candles": original_count,
                        "requested_limit": limit,
                        "unit": unit,
                        "interval": interval,
                        "first_candle": candles[0] if candles else None,
                        "last_candle": candles[-1] if candles else None
                    }
                })
                
            except Exception as e:
                errors.append({
                    "symbol": symbol,
                    "error": str(e)
                })
    else:
        # Make HTTP calls to own endpoints (useful for testing)
        base_url = "http://localhost:8000"  # Adjust this to your actual base URL
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                fetch_historical_data_async(session, base_url, symbol, limit, unit, interval)
                for symbol in symbols
            ]
            
            responses = await asyncio.gather(*tasks)
            
            for response in responses:
                if response["status"] == "success":
                    results.append(response["data"])
                else:
                    errors.append({
                        "symbol": response["symbol"],
                        "error": response["error"]
                    })
    
    return MultipleSymbolsResponse(
        processed_symbols=len(symbols),
        results=results,
        errors=errors
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

@app.get("/historical-data/{symbol}/{limit}/{interval}", response_model=CandleData, summary="Get Historical Data with Interval")
async def get_historical_data_with_interval(
    symbol: str,
    limit: int,
    interval: IntervalEnum
):
    """
    Get historical candle data for a specific NIFTY symbol with specified interval
    
    - **symbol**: Trading symbol (e.g., JIOFIN, RELIANCE)
    - **limit**: Number of most recent candlesticks to return
    - **interval**: Time interval - 1min, 15min, or 1hr
    
    Returns data with IST timestamps in 24-hour format, showing the most recent candles first.
    
    Examples:
    - /historical-data/RELIANCE/5/1min - Returns 5 most recent 1-minute candles for RELIANCE
    - /historical-data/JIOFIN/10/15min - Returns 10 most recent 15-minute candles for JIOFIN
    - /historical-data/TCS/3/1hr - Returns 3 most recent 1-hour candles for TCS
    """
    
    # Convert symbol to uppercase
    symbol = symbol.upper()
    
    # Load instruments
    instruments = load_nifty_instruments()
    
    # Find instrument key
    instrument_key = get_instrument_key(instruments, symbol)
    
    if not instrument_key:
        available_symbols = [inst.get("trading_symbol") for inst in instruments if inst.get("trading_symbol")]
        raise HTTPException(
            status_code=404, 
            detail=f"Symbol '{symbol}' not found in NIFTY instruments. Use /symbols endpoint to see available symbols."
        )
    
    # Convert interval to API parameters
    unit, api_interval = convert_interval_to_api_params(interval)
    
    # Fetch historical data
    candles = get_historical_data(instrument_key, symbol, unit, api_interval)
    
    # Filter and process candles with IST timestamps
    processed_candles = filter_candles_by_interval(candles, interval, limit)
    
    # Get current IST time for reference
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
            "unit": unit,
            "api_interval": api_interval,
            "timezone": "IST (Asia/Kolkata)",
            "timestamp_format": "YYYY-MM-DD HH:MM:SS",
            "current_ist_time": current_ist,
            "first_candle": processed_candles[0] if processed_candles else None,
            "last_candle": processed_candles[-1] if processed_candles else None
        }
    )

@app.get("/download-pre-open-csv", summary="Download Pre-Open Market CSV")
async def download_pre_open_csv():
    """Endpoint to download pre-open market data in CSV format"""
    try:
        csv_url = "https://www.nseindia.com/api/market-data-pre-open?key=NIFTY&csv=true&selectValFormat=crores"
        response = requests.get(csv_url, headers={'User-Agent': 'Mozilla/5.0'})

        if response.status_code == 200:
            return JSONResponse(
                content=response.text,
                media_type="text/csv"
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to download CSV file"
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Network error: {str(e)}")

@app.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint"""
    try:
        instruments = load_nifty_instruments()
        csv_exists = os.path.exists("data.csv")
        
        # Get current IST time
        ist = pytz.timezone('Asia/Kolkata')
        current_ist = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            "status": "healthy",
            "instruments_loaded": len(instruments),
            "nifty_json_exists": os.path.exists("NIFTY.json"),
            "data_csv_exists": csv_exists,
            "current_ist_time": current_ist,
            "supported_intervals": ["1min", "15min", "1hr"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "message": str(exc.detail)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred"}
    )


# Add these imports at the top of your existing file
import numpy as np
from typing import Tuple

# Add these new response models after your existing models
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

# Add these helper functions for technical analysis
def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return []
    
    ema = []
    multiplier = 2 / (period + 1)
    
    # Start with SMA for first EMA value
    sma = sum(prices[:period]) / period
    ema.append(sma)
    
    # Calculate EMA for remaining values
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
        
        # Update averages for next iteration
        current_gain = gains[i] if gains[i] > 0 else 0
        current_loss = losses[i] if losses[i] > 0 else 0
        
        avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
        avg_loss = ((avg_loss * (period - 1)) + current_loss) / period
    
    return rsi_values

def extract_prices_and_volumes(candles: List[List]) -> Tuple[List[float], List[float]]:
    """Extract close prices and volumes from candle data"""
    # Assuming candle format: [timestamp, open, high, low, close, volume]
    if not candles or len(candles[0]) < 6:
        return [], []
    
    prices = [float(candle[4]) for candle in candles]  # Close prices
    volumes = [float(candle[5]) for candle in candles]  # Volumes
    
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


def calculate_mvwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], period: int = 20) -> float:
    """
    Calculate Moving Volume Weighted Average Price
    
    Args:
        highs, lows, closes, volumes: OHLCV data lists
        period: Rolling window period (default 20)
    
    Returns:
        float: MVWAP value for the most recent period
    """
    if not (len(highs) == len(lows) == len(closes) == len(volumes)):
        raise ValueError("All input lists must have the same length")
    
    if len(highs) < period:
        # If not enough data, use all available data
        start_idx = 0
    else:
        # Use last 'period' bars
        start_idx = len(highs) - period
    
    cumulative_pv = 0
    cumulative_volume = 0
    
    for i in range(start_idx, len(highs)):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3
        cumulative_pv += typical_price * volumes[i]
        cumulative_volume += volumes[i]
    
    if cumulative_volume == 0:
        return 0
    
    return cumulative_pv / cumulative_volume
        
def analyze_trading_signal(candles_1m: List[List], candles_5m: List[List], candles_15m: List[List]) -> TradingSignal:
    """
    Analyze trading signal based on improved multi-timeframe strategy:
    - Primary Crossover: EMA(9) > EMA(21) on 1m timeframe
    - Trend Filter: EMA(21) > EMA(50) on 15m timeframe  
    - Higher Timeframe Trend: EMA(21) > EMA(50) on 1h timeframe
    - RSI(1m) crosses above 50
    - Volume spike > 1.5x avg volume
    """
    logger.info("Analyzing trading signal")

    if len(candles_1m) >= 30:
        high_1m, low_1m, close_1m, volume_1m = extract_hlcv(candles_1m)
        mvwap_1m_fast = calculate_mvwap(high_1m, low_1m, close_1m, volume_1m, period=10)  # Fast MVWAP
        mvwap_1m_slow = calculate_mvwap(high_1m, low_1m, close_1m, volume_1m, period=30)  # Slow MVWAP
    
    if len(candles_5m) >= 12:
        high_5m, low_5m, close_5m, volume_5m = extract_hlcv(candles_5m)
        mvwap_5m = calculate_mvwap(high_5m, low_5m, close_5m, volume_5m, period=15)
    
    if len(candles_15m) >= 8:
        high_15m, low_15m, close_15m, volume_15m = extract_hlcv(candles_15m)
        mvwap_15m = calculate_mvwap(high_15m, low_15m, close_15m, volume_15m, period=10)
    reasons = []
    technical_indicators = {}
    signal = "HOLD"
    confidence = 0.0
    
    try:
        # Extract data for all timeframes
        prices_1m, volumes_1m = extract_prices_and_volumes(candles_1m)
        logger.info(f"1m prices length: {len(prices_1m)}")
        prices_5m, volumes_5m = extract_prices_and_volumes(candles_5m)
        logger.info(f"5m prices length: {len(prices_5m)}")
        prices_15m, volumes_15m = extract_prices_and_volumes(candles_15m)
        logger.info(f"15m prices length: {len(prices_15m)}")


        
        # Check minimum data requirements
        if len(prices_1m) < 50:
            return TradingSignal(
                signal="HOLD", confidence=0.0,
                reasons=["Insufficient 1m data for analysis"],
                technical_indicators={}
            )
        
        if len(prices_5m) < 13:
            return TradingSignal(
                signal="HOLD", confidence=0.0,
                reasons=["Insufficient 5m data for analysis"],
                technical_indicators={}
            )
        
        # if len(prices_15m) < 50:
        #     return TradingSignal(
        #         signal="HOLD", confidence=0.0,
        #         reasons=["Insufficient 15m data for analysis"],
        #         technical_indicators={}
        #     )
        
        logger.info("Calculating EMA values for 1m, 5m, 15m")
        
        # Calculate EMAs for 1m
        ema5_1m = calculate_ema(prices_1m, 5)
        ema8_1m = calculate_ema(prices_1m, 8)
        ema9_1m = calculate_ema(prices_1m, 9)
        ema21_1m = calculate_ema(prices_1m, 21)
        ema26_1m = calculate_ema(prices_1m, 26)

        # Calculate EMAs for 5m
        ema3_5m = calculate_ema(prices_5m, 3)
        ema5_5m = calculate_ema(prices_5m, 5)
        ema8_5m = calculate_ema(prices_5m, 8)
        ema13_5m = calculate_ema(prices_5m, 13)
        ema21_5m = calculate_ema(prices_5m, 21)

        # Calculate RSI for 1m, 5m and 15m
        rsi_1m = calculate_rsi(prices_1m)
        # rsi_5m = calculate_rsi(prices_5m)
        # rsi_15m = calculate_rsi(prices_15m)

    
        # Check for volume spike
        avg_volume_1m = sum(volumes_1m[-20:]) / len(volumes_1m[-20:]) if len(volumes_1m) >= 20 else None
        avg_volume_5m = sum(volumes_5m[-20:]) / len(volumes_5m[-20:]) if len(volumes_5m) >= 20 else None
        avg_volume_15m = sum(volumes_15m[-20:]) / len(volumes_15m[-20:]) if len(volumes_15m) >= 20 else None

    
        # Store technical indicators
        technical_indicators = {
            "1m": {
                "current_price": prices_1m[-1],
                "ema9": ema9_1m[-1] if ema9_1m else None,
                "ema21": ema21_1m[-1] if ema21_1m else None,
                "rsi": rsi_1m[-1] if rsi_1m else None,
                "current_volume": volumes_1m[-1] if volumes_1m else None,
                "avg_volume": sum(volumes_1m[-20:]) / len(volumes_1m[-20:]) if len(volumes_1m) >= 20 else None
            },
            "5m": {
                "current_price": prices_5m[-1],
                "ema3": ema3_5m[-1] if ema3_5m else None,
                "ema5": ema5_5m[-1] if ema5_5m else None,
                "ema8": ema8_5m[-1] if ema8_5m else None,
                "ema13": ema13_5m[-1] if ema13_5m else None,
                "ema21": ema21_5m[-1] if ema21_5m else None
            }
        }
        
        # # Check buy conditions based on new strategy
        buy_conditions = []
        conditions_met = 0
        
        # Condition 1: Primary Crossover - EMA(9) > EMA(21) on 1m
        if ema9_1m and ema21_1m and ema9_1m[-1] > ema21_1m[-1]:
            buy_conditions.append("1m_primary_crossover")
            reasons.append("✅ 1m EMA9 greater than EMA21")
            conditions_met += 1
        else:
            reasons.append("❌ 1m EMA9 not above EMA21")

        
        # Condition 4: EMA(5) > EMA(9) on 1m
        if ema5_1m and ema9_1m and ema5_1m[-1] > ema9_1m[-1]:
            buy_conditions.append("1m_ema5_9")
            reasons.append("✅ 1m EMA5 greater than EMA9")
            conditions_met += 1
        else:
            reasons.append("❌ 1m EMA5 not above EMA9")
        
        # Condition 5: EMA(9) > EMA(26) on 1m
        if ema9_1m and ema26_1m and ema9_1m[-1] > ema26_1m[-1]:
            buy_conditions.append("1m_ema9_26")
            reasons.append("✅ 1m EMA9 greater than EMA26")
            conditions_met += 1
        else:
            reasons.append("❌ 1m EMA9 not above EMA26")
        
        # Condition 6: EMA(8) > EMA(21) on 1m
        if ema8_1m and ema21_1m and ema8_1m[-1] > ema21_1m[-1]:
            buy_conditions.append("1m_ema8_21")
            reasons.append("✅ 1m EMA8 greater than EMA21")
            conditions_met += 1
        else:
            reasons.append("❌ 1m EMA8 not above EMA21")

        
        # Condition 7: EMA(3) > EMA(8) on 5m
        if ema3_5m and ema8_5m and ema3_5m[-1] > ema8_5m[-1]:
            buy_conditions.append("5m_ema3_8")
            reasons.append("✅ 5m EMA3 greater than EMA8")
            conditions_met += 1
        else:
            reasons.append("❌ 5m EMA3 not above EMA8")
        
        # Condition 8: EMA(5) > EMA(13) on 5m
        if ema5_5m and ema13_5m and ema5_5m[-1] > ema13_5m[-1]:
            buy_conditions.append("5m_ema5_13")
            reasons.append("✅ 5m EMA5 greater than EMA13")
            conditions_met += 1
        else:
            reasons.append("❌ 5m EMA5 not above EMA13")
        
        # Condition 9: EMA(8) > EMA(21) on 5m
        if ema8_5m and ema21_5m and ema8_5m[-1] > ema21_5m[-1]:
            buy_conditions.append("5m_ema8_21")
            reasons.append("✅ 5m EMA8 greater than EMA21")
            conditions_met += 1
        else:
            reasons.append("❌ 5m EMA8 not above EMA21")
    
        # Condition 4: RSI(1m) crosses above 50
        if rsi_1m and rsi_1m[-1] > 50:
            if len(rsi_1m) > 1 and rsi_1m[-2] <= 50:
                buy_conditions.append("rsi_crossover")
                reasons.append("✅ RSI crossed above 50 (Strong momentum)")
                conditions_met += 1
            elif rsi_1m[-1] > 50:
                buy_conditions.append("rsi_above_50")
                reasons.append("✅ RSI above 50 (Positive momentum)")
                conditions_met += 1
        else:
            reasons.append("❌ RSI below 50")

        if volumes_1m and avg_volume_1m and volumes_1m[-1] > 1.5 * avg_volume_1m:
            reasons.append("✅ 1m volume spike greater than 1.5x avg volume")
            conditions_met += 1
        if volumes_5m and avg_volume_5m and volumes_5m[-1] > 1.5 * avg_volume_5m:
            reasons.append("✅ 5m volume spike greater than 1.5x avg volume")
            conditions_met += 1
        if volumes_15m and avg_volume_15m and volumes_15m[-1] > 1.5 * avg_volume_15m:
            reasons.append("✅ 15m volume spike greater than 1.5x avg volume")
            conditions_met += 1
        
 
        # Determine final signal with stricter criteria
        if len(buy_conditions) >= 4:  # At least 4 out of 5 conditions met
            signal = "BUY"
        elif len(buy_conditions) >= 2 and "1m_primary_crossover" in buy_conditions:
            # Must have primary crossover + at least one more condition
            signal = "HOLD"
            # confidence = min(confidence, 65)
        else:
            signal = "HOLD"
            # confidence = max(confidence, 15)

        
        confidence = (conditions_met/10)*100
        # Cap confidence at 100
        confidence = min(confidence, 100)
        
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

# Add this new endpoint to your existing FastAPI app
@app.get("/{symbol}/decide", response_model=TradingStatusResponse, summary="Get Trading Status")
async def get_trading_status(symbol: str):
    """
    Get trading recommendation (BUY/HOLD) for a symbol based on improved multi-timeframe analysis
    
    Strategy:
    - Primary Crossover: EMA(9) > EMA(21) on 1m timeframe
    - Trend Filter: EMA(21) > EMA(50) on 15m timeframe
    - Higher Timeframe Trend: EMA(21) > EMA(50) on 1h timeframe
    - RSI(1m) crosses above 50
    - Volume spike > 1.5x average volume
    """
    
    # Convert symbol to uppercase
    symbol = symbol.upper()

    logger.info("Fetching instruments from NIFTY.json file...")
    
    # Load instruments
    instruments = load_nifty_instruments()

    logger.info("Fetching instrument key from NIFTY.json...")
    
    # Find instrument key
    instrument_key = get_instrument_key(instruments, symbol)
    
    if not instrument_key:
        logger.error("Unable to find instrument key for symbol '{symbol}'")
        raise HTTPException(
            status_code=404, 
            detail=f"Symbol '{symbol}' not found in NIFTY instruments. Use /symbols endpoint to see available symbols."
        )
    
    try:
        # Fetch multi-timeframe data
        candles_1m = get_historical_data(instrument_key, symbol, "minutes", 1, 22)
        candles_5m = get_historical_data(instrument_key, symbol, "minutes", 5, 22)
        candles_15m = get_historical_data(instrument_key, symbol, "minutes", 15, 22)
        
        print(f"Fetched candles - 1m: {len(candles_1m)}, 5m: {len(candles_5m)}, 15m: {len(candles_15m)}")
        
        # Analyze trading signal with all three timeframes
        trading_signal = analyze_trading_signal(candles_1m, candles_5m, candles_15m)

        print("CANDLES: ", candles_1m)
        
        # Get current price (latest close)
        current_price = float(candles_1m[0][4]) if candles_1m else 0.0

        print("CURRENT PRICE: ", current_price)
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing trading status: {str(e)}"
        )
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

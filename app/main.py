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

def get_historical_data_with_dates(instrument_key: str, symbol: str, unit: str = "minutes", interval: int = 1, min_candles: int = 50):
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
    
    while len(all_candles) < min_candles and days_back <= max_days_back:
        # Calculate the date to fetch
        fetch_date = current_date - timedelta(days=days_back)
        to_date = fetch_date.strftime('%Y-%m-%d')
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
                    all_candles = candles + all_candles
                    print(f"Fetched {len(candles)} candles with {interval} {unit} interval for {to_date}. Total: {len(all_candles)}")
                
                days_back += 1
                
                # If we got enough candles, break
                if len(all_candles) >= min_candles:
                    break
                    
            else:
                print(f"API error for {to_date}: {response.status_code} - {response.text}")
                days_back += 1
                
        except requests.exceptions.Timeout:
            print(f"Timeout for {to_date}")
            days_back += 1
        except requests.exceptions.RequestException as e:
            print(f"Network error for {to_date}: {str(e)}")
            days_back += 1
    
    if len(all_candles) < min_candles:
        raise HTTPException(
            status_code=503, 
            detail=f"Could not fetch enough candles. Got {len(all_candles)}, need {min_candles}"
        )
    
    return all_candles

# Update the original function to use the new one
def get_historical_data(instrument_key: str, symbol: str, unit: str = "minutes", interval: int = 1):
    """Fetch historical candle data for the instrument"""
    min_candles = 100 if unit == "minutes" and interval == 1 else 50
    return get_historical_data_with_dates(instrument_key, symbol, unit, interval, min_candles)

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

def analyze_trading_signal(candles_1m: List[List], candles_15m: List[List], candles_1h: List[List]) -> TradingSignal:
    """
    Analyze trading signal based on improved multi-timeframe strategy:
    - Primary Crossover: EMA(9) > EMA(21) on 1m timeframe
    - Trend Filter: EMA(21) > EMA(50) on 15m timeframe  
    - Higher Timeframe Trend: EMA(21) > EMA(50) on 1h timeframe
    - RSI(1m) crosses above 50
    - Volume spike > 1.5x avg volume
    """
    
    reasons = []
    technical_indicators = {}
    signal = "HOLD"
    confidence = 0.0
    
    try:
        # Extract data for all timeframes
        prices_1m, volumes_1m = extract_prices_and_volumes(candles_1m)
        prices_15m, volumes_15m = extract_prices_and_volumes(candles_15m)
        prices_1h, volumes_1h = extract_prices_and_volumes(candles_1h)
        
        # Check minimum data requirements
        if len(prices_1m) < 50:
            return TradingSignal(
                signal="HOLD", confidence=0.0,
                reasons=["Insufficient 1m data for analysis"],
                technical_indicators={}
            )
        
        if len(prices_15m) < 50:
            return TradingSignal(
                signal="HOLD", confidence=0.0,
                reasons=["Insufficient 15m data for analysis"],
                technical_indicators={}
            )
            
        if len(prices_1h) < 50:
            return TradingSignal(
                signal="HOLD", confidence=0.0,
                reasons=["Insufficient 1h data for analysis"],
                technical_indicators={}
            )
        
        # Calculate EMAs for 1m
        ema9_1m = calculate_ema(prices_1m, 9)
        ema21_1m = calculate_ema(prices_1m, 21)
        
        # Calculate EMAs for 15m
        ema21_15m = calculate_ema(prices_15m, 21)
        ema50_15m = calculate_ema(prices_15m, 50)
        
        # Calculate EMAs for 1h
        ema21_1h = calculate_ema(prices_1h, 21)
        ema50_1h = calculate_ema(prices_1h, 50)
        
        # Calculate RSI for 1m
        rsi_1m = calculate_rsi(prices_1m, 14)
        
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
            "15m": {
                "current_price": prices_15m[-1],
                "ema21": ema21_15m[-1] if ema21_15m else None,
                "ema50": ema50_15m[-1] if ema50_15m else None
            },
            "1h": {
                "current_price": prices_1h[-1],
                "ema21": ema21_1h[-1] if ema21_1h else None,
                "ema50": ema50_1h[-1] if ema50_1h else None
            }
        }
        
        # Check buy conditions based on new strategy
        buy_conditions = []
        
        # Condition 1: Primary Crossover - EMA(9) > EMA(21) on 1m
        if ema9_1m and ema21_1m and ema9_1m[-1] > ema21_1m[-1]:
            buy_conditions.append("1m_primary_crossover")
            reasons.append("✅ 1m EMA9 greater than EMA21 (Primary Crossover)")
            confidence += 30
        else:
            reasons.append("❌ 1m EMA9 not above EMA21")
        
        # Condition 2: Trend Filter - EMA(21) > EMA(50) on 15m
        if ema21_15m and ema50_15m and ema21_15m[-1] > ema50_15m[-1]:
            buy_conditions.append("15m_trend_filter")
            reasons.append("✅ 15m EMA21 greater than EMA21 EMA50 (Trend Filter)")
            confidence += 25
        else:
            reasons.append("❌ 15m EMA21 not above EMA50")
        
        # Condition 3: Higher Timeframe Trend - EMA(21) > EMA(50) on 1h
        if ema21_1h and ema50_1h and ema21_1h[-1] > ema50_1h[-1]:
            buy_conditions.append("1h_trend_direction")
            reasons.append("✅ 1h EMA21 greater than EMA21 EMA50 (Higher TF Trend)")
            confidence += 20
        else:
            reasons.append("❌ 1h trend not bullish")
        
        # Condition 4: RSI(1m) crosses above 50
        if rsi_1m and rsi_1m[-1] > 50:
            if len(rsi_1m) > 1 and rsi_1m[-2] <= 50:
                buy_conditions.append("rsi_crossover")
                reasons.append("✅ RSI crossed above 50 (Strong momentum)")
                confidence += 15
            elif rsi_1m[-1] > 50:
                buy_conditions.append("rsi_above_50")
                reasons.append("✅ RSI above 50 (Positive momentum)")
                confidence += 10
        else:
            reasons.append("❌ RSI below 50")
        
        # Condition 5: Volume spike > 1.5x avg volume
        if (volumes_1m and len(volumes_1m) >= 20 and 
            technical_indicators["1m"]["current_volume"] and 
            technical_indicators["1m"]["avg_volume"]):
            
            volume_ratio = technical_indicators["1m"]["current_volume"] / technical_indicators["1m"]["avg_volume"]
            technical_indicators["1m"]["volume_ratio"] = volume_ratio
            
            if volume_ratio > 1.5:
                buy_conditions.append("volume_spike")
                reasons.append(f"✅ Volume spike detected ({volume_ratio:.2f}x avg)")
                confidence += 10
            else:
                reasons.append(f"⚠️ Normal volume ({volume_ratio:.2f}x avg)")
        else:
            reasons.append("❌ Insufficient volume data")
        
        # Determine final signal with stricter criteria
        if len(buy_conditions) >= 4:  # At least 4 out of 5 conditions met
            signal = "BUY"
        elif len(buy_conditions) >= 2 and "1m_primary_crossover" in buy_conditions:
            # Must have primary crossover + at least one more condition
            signal = "HOLD"
            confidence = min(confidence, 65)
        else:
            signal = "HOLD"
            confidence = max(confidence, 15)
        
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
    
    # Load instruments
    instruments = load_nifty_instruments()
    
    # Find instrument key
    instrument_key = get_instrument_key(instruments, symbol)
    
    if not instrument_key:
        raise HTTPException(
            status_code=404, 
            detail=f"Symbol '{symbol}' not found in NIFTY instruments. Use /symbols endpoint to see available symbols."
        )
    
    try:
        # Fetch multi-timeframe data
        candles_1m = get_historical_data(instrument_key, symbol, "minutes", 1)
        candles_15m = get_historical_data(instrument_key, symbol, "minutes", 15)
        candles_1h = get_historical_data(instrument_key, symbol, "hours", 1)
        
        print(f"Fetched candles - 1m: {len(candles_1m)}, 15m: {len(candles_15m)}, 1h: {len(candles_1h)}")
        
        # Analyze trading signal with all three timeframes
        trading_signal = analyze_trading_signal(candles_1m, candles_15m, candles_1h)
        
        # Get current price (latest close)
        current_price = float(candles_1m[-1][4]) if candles_1m else 0.0
        
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

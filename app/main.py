from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import pandas as pd
from datetime import datetime, timedelta
import os

app = FastAPI(title="Candlestick Data API", version="1.0.0")

# Twelve Data API key - get free one from twelvedata.com
TWELVE_DATA_API_KEY = "84cc761328cc48ccbe48723d717f974a"

if not TWELVE_DATA_API_KEY:
    print("WARNING: TWELVE_DATA_API_KEY environment variable not set!")
    print("Get your free API key from: https://twelvedata.com/account/api")

# Response Models
class Candlestick(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class CandlestickResponse(BaseModel):
    symbol: str
    interval: str
    data: List[Candlestick]
    last_updated: str

# Helper function to fetch data from Twelve Data
async def fetch_intraday_data(symbol: str, interval: str, period_days: int) -> pd.DataFrame:
    print(f"DEBUG: Fetching intraday data for {symbol} with interval {interval} for {period_days} days...")
    
    if not TWELVE_DATA_API_KEY:
        print("ERROR: No Twelve Data API key found")
        raise HTTPException(status_code=500, detail="Twelve Data API key not configured. Get free key from twelvedata.com")
    
    # Map intervals to Twelve Data format
    interval_map = {
        "1m": "1min",
        "15m": "15min", 
        "1h": "1h"
    }
    
    if interval not in interval_map:
        print(f"ERROR: Unsupported interval: {interval}")
        raise HTTPException(status_code=400, detail=f"Unsupported interval: {interval}")
    
    twelve_interval = interval_map[interval]
    
    # Construct Twelve Data API URL for time series
    url = "https://api.twelvedata.com/time_series"
    
    # Calculate outputsize based on interval and period
    if interval == "1m":
        outputsize = min(390 * period_days, 5000)  # 390 minutes per trading day, max 5000
    elif interval == "15m":
        outputsize = min(26 * period_days, 5000)   # 26 15-min intervals per trading day
    else:  # 1h
        outputsize = min(7 * period_days, 5000)    # ~7 hours per trading day
    
    params = {
        "symbol": symbol.upper(),
        "interval": twelve_interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }
    
    print(f"DEBUG: Making request to Twelve Data API")
    print(f"DEBUG: URL: {url}")
    print(f"DEBUG: Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"DEBUG: Response status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"ERROR: Twelve Data API returned status {response.status_code}")
            print(f"ERROR: Response text: {response.text}")
            raise HTTPException(status_code=503, detail=f"Twelve Data API error: {response.status_code}")
        
        data = response.json()
        print(f"DEBUG: Response JSON keys: {list(data.keys())}")
        
        # Check for API errors
        if "code" in data and data["code"] != 200:
            print(f"ERROR: Twelve Data API error: {data.get('message', 'Unknown error')}")
            print(f"ERROR: Full response: {data}")
            if "Invalid API key" in str(data.get('message', '')):
                raise HTTPException(status_code=401, detail="Invalid API key. Get free key from twelvedata.com")
            raise HTTPException(status_code=503, detail=f"API error: {data.get('message', 'Unknown error')}")
        
        # Check if we have values
        if "values" not in data or not data["values"]:
            print(f"ERROR: No values found for symbol: {symbol}")
            print(f"ERROR: Response: {data}")
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
        
        values = data["values"]
        print(f"DEBUG: Number of data points: {len(values)}")
        
        # Convert to DataFrame
        df_data = []
        for item in values:
            try:
                timestamp = datetime.strptime(item["datetime"], "%Y-%m-%d %H:%M:%S")
                df_data.append({
                    "Open": float(item["open"]),
                    "High": float(item["high"]), 
                    "Low": float(item["low"]),
                    "Close": float(item["close"]),
                    "Volume": int(item["volume"]),
                    "Timestamp": timestamp
                })
                print(f"DEBUG: Processed data point: {timestamp}")
            except Exception as e:
                print(f"ERROR: Error processing data point {item}: {str(e)}")
                continue
        
        if not df_data:
            print(f"ERROR: No valid data points processed for symbol: {symbol}")
            raise HTTPException(status_code=404, detail=f"No valid data found for symbol: {symbol}")
        
        df = pd.DataFrame(df_data)
        df.set_index("Timestamp", inplace=True)
        df.sort_index(inplace=True)  # Sort by timestamp
        
        print(f"DEBUG: Created DataFrame with {len(df)} rows")
        print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
        print(f"DEBUG: DataFrame index (first 3): {df.index[:3].tolist()}")
        print(f"DEBUG: DataFrame index (last 3): {df.index[-3:].tolist()}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request exception: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Network error: {str(e)}")
    except Exception as e:
        print(f"ERROR: Unexpected error fetching data: {str(e)}")
        raise HTTPException(status_code=503, detail=f"External API error: {str(e)}")

def process_candlestick_data(data: pd.DataFrame, symbol: str, interval: str) -> CandlestickResponse:
    """
    Process raw Twelve Data into structured candlestick format
    """
    print("DEBUG: Processing candlestick data...")
    print(f"DEBUG: Data shape: {data.shape}")
    print(f"DEBUG: Data columns: {data.columns.tolist()}")
    
    if data.empty:
        print("ERROR: DataFrame is empty")
        raise HTTPException(status_code=404, detail="No data found for the given symbol")
    
    candlesticks = []
    
    # Process data (most recent first)
    for timestamp, row in data.iterrows():
        try:
            print(f"DEBUG: Processing row for timestamp {timestamp}")
            candlestick = Candlestick(
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                open=round(float(row['Open']), 2),
                high=round(float(row['High']), 2),
                low=round(float(row['Low']), 2),
                close=round(float(row['Close']), 2),
                volume=int(row['Volume'])
            )
            candlesticks.append(candlestick)
            print(f"DEBUG: Successfully created candlestick: {candlestick.timestamp}")
        except Exception as e:
            print(f"ERROR: Error processing row {timestamp}: {str(e)}")
            continue
    
    # Reverse to get most recent first
    candlesticks.reverse()
    print(f"DEBUG: Reversed candlesticks, most recent first")
    
    # Get last updated time
    last_updated = datetime.now().isoformat()
    
    print(f"DEBUG: Processed {len(candlesticks)} candlesticks")
    
    return CandlestickResponse(
        symbol=symbol.upper(),
        interval=interval,
        data=candlesticks,
        last_updated=last_updated
    )

@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Candlestick Data API (Twelve Data)",
        "version": "1.0.0",
        "endpoints": [
            "/candlestick/1min/{symbol}",
            "/candlestick/15min/{symbol}",
            "/candlestick/1hour/{symbol}"
        ],
        "data_source": "Twelve Data",
        "note": "Free tier: 800 API calls/day. Get free API key from twelvedata.com"
    }

@app.get("/candlestick/1min/{symbol}", response_model=CandlestickResponse)
async def get_1min_candlesticks(symbol: str):
    """
    Get 1-minute candlestick data for a given symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL, GOOGL, MSFT)
    
    Returns:
        CandlestickResponse with 1-minute interval data
    """
    print(f"DEBUG: Endpoint /candlestick/1min/{symbol} called")
    data = await fetch_intraday_data(symbol, "1m", 1)  # 1 day of 1-minute data
    return process_candlestick_data(data, symbol, "1min")

@app.get("/candlestick/15min/{symbol}", response_model=CandlestickResponse)
async def get_15min_candlesticks(symbol: str):
    """
    Get 15-minute candlestick data for a given symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL, GOOGL, MSFT)
    
    Returns:
        CandlestickResponse with 15-minute interval data
    """
    print(f"DEBUG: Endpoint /candlestick/15min/{symbol} called")
    data = await fetch_intraday_data(symbol, "15m", 5)  # 5 days of 15-minute data
    return process_candlestick_data(data, symbol, "15min")

@app.get("/candlestick/1hour/{symbol}", response_model=CandlestickResponse)
async def get_1hour_candlesticks(symbol: str):
    """
    Get 1-hour candlestick data for a given symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL, GOOGL, MSFT)
    
    Returns:
        CandlestickResponse with 1-hour interval data
    """
    print(f"DEBUG: Endpoint /candlestick/1hour/{symbol} called")
    data = await fetch_intraday_data(symbol, "1h", 30)  # 30 days of 1-hour data
    return process_candlestick_data(data, symbol, "1hour")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    print("DEBUG: Health check endpoint called")
    api_status = "configured" if TWELVE_DATA_API_KEY else "not configured"
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "api_key_status": api_status
    }

# Get available intervals
@app.get("/intervals")
async def get_available_intervals():
    """
    Get list of available candlestick intervals
    """
    print("DEBUG: Intervals endpoint called")
    return {
        "available_intervals": [
            {"interval": "1min", "description": "1-minute candlesticks (1 day of data)"},
            {"interval": "15min", "description": "15-minute candlesticks (5 days of data)"},
            {"interval": "1hour", "description": "1-hour candlesticks (30 days of data)"}
        ],
        "data_source": "Twelve Data",
        "rate_limits": "Free tier: 800 API calls/day",
        "note": "Get free API key from https://twelvedata.com/account/api"
    }

# Test endpoint to verify API connectivity
@app.get("/test/{symbol}")
async def test_api_connection(symbol: str):
    """
    Test endpoint to verify Twelve Data API connectivity
    """
    print(f"DEBUG: Test endpoint called for symbol: {symbol}")
    
    if not TWELVE_DATA_API_KEY:
        return {
            "symbol": symbol.upper(),
            "api_status": "no_api_key",
            "error": "TWELVE_DATA_API_KEY not set",
            "note": "Get free API key from https://twelvedata.com/account/api"
        }
    
    url = "https://api.twelvedata.com/quote"
    params = {
        "symbol": symbol.upper(),
        "apikey": TWELVE_DATA_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        print(f"DEBUG: Test API response: {data}")
        
        if "code" in data and data["code"] != 200:
            return {
                "symbol": symbol.upper(),
                "api_status": "api_error",
                "error": data.get("message", "Unknown error"),
                "response": data
            }
        
        return {
            "symbol": symbol.upper(),
            "api_status": "connected",
            "current_price": data.get("close", "N/A"),
            "response": data,
            "api_key_type": "configured"
        }
    except Exception as e:
        print(f"ERROR: Test API failed: {str(e)}")
        return {
            "symbol": symbol.upper(),
            "api_status": "failed",
            "error": str(e),
            "api_key_type": "configured"
        }

# Endpoint to check API usage/credits
@app.get("/usage")
async def check_api_usage():
    """
    Check API usage (if supported by Twelve Data)
    """
    print("DEBUG: Usage endpoint called")
    
    if not TWELVE_DATA_API_KEY:
        return {
            "status": "no_api_key",
            "message": "API key not configured"
        }
    
    # Twelve Data doesn't have a direct usage endpoint, but we can test with a simple quote
    url = "https://api.twelvedata.com/quote"
    params = {
        "symbol": "AAPL",  # Use a common symbol for testing
        "apikey": TWELVE_DATA_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "code" in data and data["code"] != 200:
            return {
                "status": "api_error",
                "message": data.get("message", "Unknown error"),
                "note": "This might indicate API limit exceeded or invalid key"
            }
        
        return {
            "status": "api_working",
            "message": "API key is functional",
            "free_tier_limit": "800 calls per day",
            "note": "Twelve Data doesn't provide usage statistics in free tier"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    print("DEBUG: Starting FastAPI application...")
    print(f"DEBUG: Twelve Data API key configured: {'Yes' if TWELVE_DATA_API_KEY else 'No'}")
    print("DEBUG: Get your free API key from: https://twelvedata.com/account/api")
    print("DEBUG: Free tier includes 800 API calls per day with intraday data support!")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

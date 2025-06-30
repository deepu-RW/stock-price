from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from urllib.parse import quote

app = FastAPI(title="Candlestick Data API", version="1.0.0")

# Configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "3N4OT147KXKHUZOR")
BASE_URL = "https://www.alphavantage.co/query"

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

# Helper function to fetch data from Alpha Vantage
async def fetch_intraday_data(symbol: str, interval: str) -> dict:
    print(f"Fetching intraday data for {symbol} with interval {interval}...")
    """
    Fetch intraday data from Alpha Vantage API
    """
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol.upper(),
        "interval": interval,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact"  # Last 100 data points
    }
    
    try:
        print(f"Requesting URL: {BASE_URL} with params: {params}")
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            error_message = data["Error Message"]
            print(f"API Error: {error_message}")
            raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
        
        if "Note" in data:
            raise HTTPException(status_code=429, detail="API call frequency limit reached")
            
        return data
    
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"External API error: {str(e)}")

def process_candlestick_data(data: dict, symbol: str, interval: str) -> CandlestickResponse:
    """
    Process raw API data into structured candlestick format
    """
    time_series_key = f"Time Series ({interval})"
    
    if time_series_key not in data:
        raise HTTPException(status_code=404, detail="No data found for the given symbol")
    
    time_series = data[time_series_key]
    candlesticks = []
    
    # Sort by timestamp (most recent first)
    sorted_timestamps = sorted(time_series.keys(), reverse=True)
    
    for timestamp in sorted_timestamps:
        candle_data = time_series[timestamp]
        
        candlestick = Candlestick(
            timestamp=timestamp,
            open=float(candle_data["1. open"]),
            high=float(candle_data["2. high"]),
            low=float(candle_data["3. low"]),
            close=float(candle_data["4. close"]),
            volume=int(candle_data["5. volume"])
        )
        candlesticks.append(candlestick)
    
    # Get last refreshed time
    last_updated = data.get("Meta Data", {}).get("3. Last Refreshed", "Unknown")
    
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
        "message": "Candlestick Data API",
        "version": "1.0.0",
        "endpoints": [
            "/candlestick/1min/{symbol}",
            "/candlestick/15min/{symbol}",
            "/candlestick/1hour/{symbol}"
        ]
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
    data = await fetch_intraday_data(symbol, "1min")
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
    data = await fetch_intraday_data(symbol, "15min")
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
    data = await fetch_intraday_data(symbol, "60min")
    return process_candlestick_data(data, symbol, "60min")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Get available intervals
@app.get("/intervals")
async def get_available_intervals():
    """
    Get list of available candlestick intervals
    """
    return {
        "available_intervals": [
            {"interval": "1min", "description": "1-minute candlesticks"},
            {"interval": "15min", "description": "15-minute candlesticks"},
            {"interval": "1hour", "description": "1-hour candlesticks"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

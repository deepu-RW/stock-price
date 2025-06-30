from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

app = FastAPI(title="Candlestick Data API", version="1.0.0")

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

# Helper function to fetch data from Yahoo Finance
async def fetch_intraday_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    print(f"Fetching intraday data for {symbol} with interval {interval}...")
    """
    Fetch intraday data from Yahoo Finance API
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        print(f"Requesting data for {symbol} with interval {interval} and period {period}")
        
        # Download historical data
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            print(f"No data found for symbol: {symbol}")
            raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
            
        print(f"Successfully fetched {len(data)} data points")
        return data
    
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise HTTPException(status_code=503, detail=f"External API error: {str(e)}")

def process_candlestick_data(data: pd.DataFrame, symbol: str, interval: str) -> CandlestickResponse:
    """
    Process raw Yahoo Finance data into structured candlestick format
    """
    print("Processing candlestick data...")
    print("Data columns:", data.columns.tolist())
    
    if data.empty:
        raise HTTPException(status_code=404, detail="No data found for the given symbol")
    
    candlesticks = []
    
    # Process data (most recent first)
    for timestamp, row in data.iterrows():
        try:
            candlestick = Candlestick(
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                open=round(float(row['Open']), 2),
                high=round(float(row['High']), 2),
                low=round(float(row['Low']), 2),
                close=round(float(row['Close']), 2),
                volume=int(row['Volume'])
            )
            candlesticks.append(candlestick)
        except Exception as e:
            print(f"Error processing row {timestamp}: {str(e)}")
            continue
    
    # Reverse to get most recent first
    candlesticks.reverse()
    
    # Get last updated time
    last_updated = datetime.now().isoformat()
    
    print(f"Processed {len(candlesticks)} candlesticks")
    
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
        "message": "Candlestick Data API (Yahoo Finance)",
        "version": "1.0.0",
        "endpoints": [
            "/candlestick/1min/{symbol}",
            "/candlestick/15min/{symbol}",
            "/candlestick/1hour/{symbol}"
        ],
        "data_source": "Yahoo Finance (yfinance)"
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
    data = await fetch_intraday_data(symbol, "1m", "1d")  # 1 day of 1-minute data
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
    data = await fetch_intraday_data(symbol, "15m", "5d")  # 5 days of 15-minute data
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
    data = await fetch_intraday_data(symbol, "1h", "30d")  # 30 days of 1-hour data
    return process_candlestick_data(data, symbol, "1hour")

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
            {"interval": "1min", "description": "1-minute candlesticks (1 day of data)"},
            {"interval": "15min", "description": "15-minute candlesticks (5 days of data)"},
            {"interval": "1hour", "description": "1-hour candlesticks (30 days of data)"}
        ],
        "data_source": "Yahoo Finance",
        "note": "No API key required"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

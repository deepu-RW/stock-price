from fastapi import FastAPI, HTTPException
from datetime import datetime
from pathlib import Path
import uuid
import pytz
import asyncio
import sys
import uvicorn
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from enums import (
    IntervalEnum,
    CandleData,
    AvailableSymbolsResponse,
    TradingStatusResponse,
)
from utils import (
    get_historical_data,
    load_nifty_instruments,
    analyze_trading_signal,
    download_nse_data,
    convert_interval_to_api_params,
    filter_candles_by_interval
)


DOWNLOAD_BASE_PATH = Path("downloads")

# Configure logging
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="ERROR")

DOWNLOAD_BASE_PATH = Path("downloads")
if not DOWNLOAD_BASE_PATH.exists():
    DOWNLOAD_BASE_PATH.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="NIFTY50 Buy/Sell/Hold Predictor",
    description="Intraday API to decide whether to buy, sell or hold a stock based on historical data",
    version="1.0.0"
)
@app.get("/", summary="API Information")
async def root():
    """Root endpoint with comprehensive API information"""
    return {
        "message": "NIFTY Buy/Sell/Hold Predictor",
        "version": "2.0.0",
        "features": [
            "Automated NSE pre-open data download",
            "Multi-timeframe technical analysis",
            "Multiple Buy/Sell Indicators",
            "Real-time market data processing"
        ],
        "endpoints": {
            "/download": "Download latest NSE pre-open data",
            "/symbols": "Get all available NIFTY symbols",
            "/{symbol}/{limit}/{interval}": "Get historical data",
            "/{symbol}/decide": "Get trading recommendation",
        },
        "supported_intervals": ["1min", "5min"],
        "timestamp_format": "IST 24-hour format (YYYY-MM-DD HH:MM:SS)"
    }

@app.get("/symbols", response_model=AvailableSymbolsResponse, summary="Get Available Symbols")
async def get_available_symbols():
    """Get all available NIFTY symbols from the CSV file"""
    try:
        instruments = load_nifty_instruments()
    except FileNotFoundError as e:
        logger.error(f"Error fetching available symbols: {str(e)}")
        return AvailableSymbolsResponse(
            total_symbols=0,
            symbols=[]
        )

    return AvailableSymbolsResponse(
        total_symbols=len(instruments),
        symbols=instruments
    )

@app.get("/{symbol}/{limit}/{interval}", response_model=CandleData, summary="Get Historical Data")
async def get_historical_data_with_interval(
    symbol: str,
    limit: int,
    interval: IntervalEnum
):
    """Get historical candle data for a specific NIFTY symbol with specified interval"""
    
    symbol = symbol.upper()
    df = pd.read_csv('https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz')

    instrument_key = df.loc[(df.tradingsymbol == symbol) & (df.exchange == 'NSE_EQ'), 'instrument_key'].values[0]

    logger.info(f"Instrument key for {symbol} is {instrument_key}")
    
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
    symbol = symbol.upper()
    logger.info(f"Analyzing trading status for {symbol}")
    
    df = pd.read_csv('https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz')

    instrument_key = df.loc[(df.tradingsymbol == symbol) & (df.exchange == 'NSE_EQ'), 'instrument_key'].values[0]

    logger.info(f"Instrument key for {symbol} is {instrument_key}")
    
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

@app.get("/download")
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
            "file_name": result["file_name"]
        }
        
        # Add information about replaced files if any
        if result.get("replaced_files"):
            response_data["replaced_files"] = result["replaced_files"]
            response_data["message"] += f" (replaced {len(result['replaced_files'])} existing CSV file(s))"
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
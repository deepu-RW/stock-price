from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import requests
import json
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

app = FastAPI(
    title="NIFTY Historical Data API",
    description="API to fetch historical candle data for NIFTY instruments",
    version="1.0.0"
)

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

# Global variable to cache instruments
_instruments_cache = None

def load_nifty_instruments():
    """Load the filtered NIFTY instruments from NIFTY.json"""
    global _instruments_cache
    
    if _instruments_cache is not None:
        return _instruments_cache
    
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "NIFTY.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"NIFTY.json not found at {json_path}")
            
        with open(json_path, "r") as file:
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

def get_historical_data(instrument_key: str, symbol: str, unit: str = "minutes", interval: int = 1):
    """Fetch historical candle data for the instrument"""
    
    payload = {}
    headers = {
        'Accept': 'application/json'
    }

    # Build the URL
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"

    try:
        # Make the request
        response = requests.get(url, headers=headers, data=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json().get("data", {})
            candles = data.get('candles', [])
            return candles
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Upstox API error: {response.text}"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Request timeout while fetching data from Upstox API")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Network error: {str(e)}")

@app.get("/", summary="API Information")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NIFTY Historical Data API",
        "version": "1.0.0",
        "endpoints": {
            "/symbols": "Get all available NIFTY symbols",
            "/historical-data/{symbol}": "Get historical data for a specific symbol",
            "/docs": "Interactive API documentation"
        }
    }

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

@app.get("/historical-data/{symbol}", response_model=CandleData, summary="Get Historical Data")
async def get_historical_data_endpoint(
    symbol: str,
    unit: str = Query(default="minutes", description="Time unit (minutes, hours, days)"),
    interval: int = Query(default=1, description="Interval value", ge=1)
):
    """
    Get historical candle data for a specific NIFTY symbol
    
    - **symbol**: Trading symbol (e.g., JIOFIN, RELIANCE)
    - **unit**: Time unit - minutes, hours, or days (default: minutes)
    - **interval**: Interval value (default: 1)
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
    
    # Fetch historical data
    candles = get_historical_data(instrument_key, symbol, unit, interval)
    
    return CandleData(
        symbol=symbol,
        instrument_key=instrument_key,
        candles=candles,
        metadata={
            "total_candles": len(candles),
            "unit": unit,
            "interval": interval,
            "first_candle": candles[0] if candles else None,
            "last_candle": candles[-1] if candles else None
        }
    )

@app.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "NIFTY.json")
        
        instruments = load_nifty_instruments()
        return {
            "status": "healthy",
            "instruments_loaded": len(instruments),
            "nifty_json_exists": os.path.exists(json_path),
            "json_path": json_path,
            "current_working_directory": os.getcwd(),
            "script_directory": current_dir
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

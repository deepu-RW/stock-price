from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

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
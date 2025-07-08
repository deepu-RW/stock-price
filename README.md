# 📈 NIFTY Buy/Sell/Hold Predictor API

This API predicts **Buy / Hold / Sell** decisions for **NIFTY 50** stocks using multi-timeframe technical analysis and real-time market data.

---

## 🚀 Features

- ✅ Automated **NSE Pre-Open Data** download  
- 🔄 Multi-timeframe **technical indicator analysis**  
- 📊 Real-time **Buy/Sell signal evaluation**  
- ⏱️ Live data processing with **1min** and **5min** interval support  

---

## 📡 API Endpoints

### `GET /`
Root endpoint with project metadata and usage details.

Returns:
```json
{
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
    "/{symbol}/decide": "Get trading recommendation"
  },
  "supported_intervals": ["1min", "5min"],
  "timestamp_format": "IST 24-hour format (YYYY-MM-DD HH:MM:SS)"
}
```

---

### `GET /download`
Downloads the latest **NSE pre-open market data** and caches it for analysis.

---

### `GET /symbols`
Returns a list of all supported **NIFTY 50** stock symbols available for prediction.

---

### `GET /{symbol}/{limit}/{interval}`
Fetches historical OHLCV data for a NIFTY stock.

**Path Parameters:**
- `symbol` – NIFTY stock symbol (e.g., `RELIANCE`)
- `limit` – Number of recent data points (e.g., `50`)
- `interval` – Candlestick interval (`1min` or `5min`)

**Example:**  
```
/RELIANCE/50/5min
```

---

### `GET /{symbol}/decide`
Returns a real-time **Buy / Hold / Sell** recommendation based on technical indicators.

**Example:**  
```
/INFY/decide
```

---

## 📌 Supported Intervals

- `1min`
- `5min`

> All timestamps are in **IST** (`YYYY-MM-DD HH:MM:SS`) format.

---

## ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/nifty-predictor-api.git

# Install dependencies
pip install -r requirements.txt

# Run the API (for development)
cd app
uvicorn main:app --reload
```

---

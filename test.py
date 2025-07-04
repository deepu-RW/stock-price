# import pytz
# from datetime import datetime, timedelta
# ist = pytz.timezone('Asia/Kolkata')
# current_date = datetime.now(ist)
# from loguru import logger
# import sys
# logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="DEBUG")
# logger.debug("DEBUG")
# logger.info("INFO")
# logger.warning("WARNING")
# logger.error("ERROR")
import requests
import pytz
from datetime import datetime

payload = {}
headers = {
    'Accept': 'application/json'
}

# Get current date in IST
ist = pytz.timezone('Asia/Kolkata')
current_date = datetime.now(ist)
instrument_key = "NSE_EQ|INE040A01034"
unit = "minutes"
interval = 5

all_candles = []



print("Current date: ", current_date.strftime('%Y-%m-%d'))
url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
params = {}
# params = {
#             'from_date': "2025-07-01",
#             'to_date': "2025-07-03"
# }
        

response = requests.get(url, headers=headers)

print("response: ", response.content)

if response.status_code == 200:
    data = response.json().get("data", {})
    print("DATA: ", data)
    candles = data.get('candles', [])
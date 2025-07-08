import pandas as pd  
df = pd.read_csv('https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz')
# df = pd.read_csv('instruments.csv')
instrument_key = df.loc[(df.tradingsymbol == 'JIOFIN') & (df.exchange == 'NSE_EQ'), 'instrument_key']

print(instrument_key.values[0])


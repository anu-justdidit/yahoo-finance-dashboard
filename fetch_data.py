"""
fetch_data.py
Fetch multiple tickers and save to data/ folder.
"""

import yfinance as yf
import os

os.makedirs('data', exist_ok=True)

tickers = ['^GSPC', 'AAPL', 'TSLA', 'GOOG', 'MSFT', 'BTC-USD', 'ETH-USD' ]

for ticker in tickers:
    print(f"Fetching {ticker}...")
    df = yf.download(ticker, start='2010-01-01')
    df.to_csv(f"data/{ticker}.csv")
    print(f"Saved to data/{ticker}.csv ✅")

print(" All data fetched!")


#!/usr/bin/env python3
# PROPRIETARY CODE - © 2024 [ANUSHA  SAHA]. All rights reserved.
# Unauthorized copying, distribution, or use is strictly prohibited.
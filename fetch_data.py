"""
fetch_data.py
Fetch multiple tickers and save to data/ folder with error handling and progress tracking.
"""

import yfinance as yf
import os
import pandas as pd
import time
from datetime import datetime

# Create data directory
os.makedirs('data', exist_ok=True)

tickers = ['^GSPC', 'AAPL', 'TSLA', 'GOOG', 'MSFT', 'BTC-USD', 'ETH-USD']

def fetch_ticker_data(ticker, retries=3):
    """Fetch data for a single ticker with retry logic"""
    for attempt in range(retries):
        try:
            print(f"Fetching {ticker} (attempt {attempt + 1}/{retries})...")
            df = yf.download(ticker, start='2010-01-01', progress=False)
            
            if df.empty:
                print(f"No data for {ticker}, retrying...")
                time.sleep(2)
                continue
                
            # Reset index to make Date a column
            df = df.reset_index()
            # Format date consistently
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            df.to_csv(f"data/{ticker}.csv", index=False)
            print(f"Saved to data/{ticker}.csv ✅ ({len(df)} rows)")
            return True
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            time.sleep(3)
    
    print(f"Failed to fetch {ticker} after {retries} attempts")
    return False

def main():
    print("Starting data fetch process...")
    print(f"Tickers to fetch: {', '.join(tickers)}")
    print("-" * 50)
    
    success_count = 0
    for ticker in tickers:
        if fetch_ticker_data(ticker):
            success_count += 1
        print()  # Empty line for readability
    
    print("-" * 50)
    print(f"Process completed! {success_count}/{len(tickers)} tickers fetched successfully")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
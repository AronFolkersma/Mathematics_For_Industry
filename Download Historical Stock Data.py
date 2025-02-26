import yfinance as yf
import pandas as pd
import time
import os

# List of Dow Jones stocks
stocks = ["AAPL", "NVDA", "MSFT", "AMZN", "WMT", "JPM", "V", "UNH", "PG", 
          "JNJ", "HD", "KO", "CRM", "CVX", "CSCO", "IBM", "MRK", "MCD", "AXP", 
          "GS", "DIS", "VZ", "CAT", "AMGN", "HON", "BA", "NKE", "SHW", "MMM",
          "TRV"]

# Define time period
start_date = "2009-01-01"
end_date = "2025-01-01"

file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "DATA")

for stock_symbol in stocks:
    historical_data = yf.download(stock_symbol).reset_index()
    time.sleep(1)
    historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.date
    
    min_date = pd.to_datetime(start_date).date()
    max_date = pd.to_datetime(end_date).date()
    
    historical_data = historical_data[historical_data['Date'] >= min_date].reset_index(drop = True)
    historical_data = historical_data[historical_data['Date'] < max_date].reset_index(drop = True)
    
    date = historical_data["Date"]
    close = historical_data[["Close"]].squeeze()
    data = pd.DataFrame({"Date": date, "Close": close})
    
    csv_file_path = os.path.join(data_path, "DATA_" + stock_symbol + ".csv")
    data.to_csv(csv_file_path, index=False)

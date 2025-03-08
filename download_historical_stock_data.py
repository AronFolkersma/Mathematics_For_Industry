import yfinance as yf
import pandas as pd
import time
import os

# List of S&P500 stocks
snp500_stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]["Symbol"]

# Define time period
start_date = "2009-01-01"
end_date = "2025-01-01"

data_path = os.path.join(os.path.expanduser('~'), 'Downloads', 
                         'Mathematics_For_Industry-main/Mathematics_For_Industry-main/DATA')

stocks = ['MMM', 'AOS', 'ABT', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALB', 'ARE', 'ALGN', 'LNT', 'ALL', 'GOOGL', 'MO', 'AMZN', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'AAPL', 'AMAT', 'ACGL', 'ADM', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BWA', 'BSX', 'BMY', 'BR', 'BRO', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CAT', 'CBRE', 'CE', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CSGP', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DECK', 'DE', 'DAL', 'DVN', 'DXCM', 'DLR', 'DFS', 'DLTR', 'D', 'DPZ', 'DOV', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ETR', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'FMC', 'F', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEN', 'GD', 'GIS', 'GPC', 'GILD', 'GPN', 'GL', 'GS', 'HAL', 'HIG', 'HAS', 'HSIC', 'HSY', 'HES', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KDP', 'KEY', 'KMB', 'KIM', 'KLAC', 'KR', 'LHX', 'LH', 'LRCX', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'MTB', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'PCAR', 'PKG', 'PARA', 'PH', 'PAYX', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'SHW', 'SPG', 'SWKS', 'SJM', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VTRS', 'V', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WMB', 'WTW', 'WYNN', 'XEL', 'YUM', 'ZBRA', 'ZBH']
ESG = [42.86, 26.82, 21.62, 11.09, 14.06, 12.51, 24.32, 18.58, 11.31, 14.83, 13.31, 20.11, 13.14, 18.58, 17.11, 24.68, 24.89, 27.88, 26.1, 26.97, 21.95, 18.28, 23.6, 12.62, 19.18, 18.71, 14.23, 22.76, 18.02, 16.7, 14.69, 15.29, 43.08, 16.36, 11.56, 20.27, 33.7, 20.26, 23.48, 22.07, 31.23, 15.14, 15.12, 10.98, 8.09, 10.98, 30.49, 19.12, 13.06, 24.36, 21.91, 23.67, 13.67, 25.97, 20.51, 23.32, 25.17, 19.93, 36.55, 14.76, 9.93, 22.06, 21.2, 18.18, 21.11, 26.73, 32.92, 11.99, 17.44, 14.49, 14.18, 26.3, 20.42, 11.32, 11.08, 21.94, 28.33, 8.96, 24.96, 12.43, 15.29, 24.96, 28.16, 18.14, 23.3, 38.36, 22.6, 23.3, 22.65, 12.99, 23.11, 17.03, 12.91, 21.89, 20.03, 17.08, 20.26, 24.16, 15.34, 24.12, 22.27, 26.94, 33.14, 21.11, 26.39, 14.69, 18.25, 17.5, 21.68, 29.12, 32.75, 11.98, 21.07, 22.82, 18.27, 8.52, 27.46, 21.78, 14.23, 16.2, 30.38, 31.55, 20.24, 12.63, 22.81, 18.8, 28.58, 27.49, 24.45, 21.64, 31.28, 27.16, 26.87, 34.47, 21.05, 15.2, 23.86, 23.97, 22.02, 12.96, 9.31, 22.79, 25.02, 34.38, 31.69, 22.21, 13.14, 11.43, 27.91, 11.61, 23.96, 17.96, 29.24, 18.08, 18.98, 21.4, 18.4, 13.67, 43.66, 16.3, 15.51, 20.09, 25.04, 12.44, 19.14, 18.24, 16.87, 17.31, 27.51, 19.21, 25.08, 27.57, 19.67, 28.1, 20.99, 16.67, 32.49, 14.61, 34.09, 26.0, 12.83, 21.68, 19.63, 20.43, 25.22, 23.84, 16.78, 7.28, 14.5, 24.98, 31.97, 21.85, 12.61, 27.12, 26.11, 12.86, 10.59, 17.94, 18.9, 16.38, 13.27, 30.65, 16.32, 26.26, 23.59, 21.37, 19.17, 18.24, 23.17, 22.22, 8.89, 16.82, 18.03, 21.61, 14.25, 14.34, 9.45, 18.66, 23.89, 20.1, 15.92, 27.3, 13.28, 30.44, 24.29, 25.21, 24.45, 11.22, 13.08, 22.8, 24.2, 19.74, 12.21, 20.65, 17.26, 26.02, 22.38, 23.62, 11.63, 17.24, 10.99, 28.14, 28.56, 11.93, 14.11, 25.26, 13.87, 19.32, 19.98, 23.9, 22.27, 16.13, 16.89, 26.04, 25.56, 13.06, 22.16, 19.96, 15.09, 10.73, 24.27, 29.56, 18.6, 13.51, 11.68, 13.74, 22.84, 27.98, 21.98, 19.03, 32.89, 14.62, 24.82, 37.28, 17.87, 14.4, 13.63, 14.34, 15.6, 21.39, 25.05, 18.44, 20.58, 24.15, 23.04, 24.92, 26.61, 34.25, 31.93, 12.23, 21.0, 11.78, 37.83, 15.93, 14.57, 21.27, 26.37, 14.89, 31.38, 17.64, 13.64, 26.84, 16.67, 22.0, 19.98, 19.36, 30.51, 26.55, 25.96, 23.71, 10.93, 22.3, 26.91, 11.18, 24.93, 19.83, 10.93, 18.63, 25.34, 18.48, 13.08, 21.13, 36.6, 13.62, 20.49, 13.91, 26.68, 29.84, 15.53, 11.31, 18.39, 14.3, 17.04, 24.67, 15.58, 20.36, 18.6, 19.42, 17.03, 18.45, 11.45, 15.18, 9.73, 18.48, 11.55, 23.22, 29.38, 12.49, 24.11, 27.19, 30.88, 28.14, 28.41, 26.16, 22.34, 22.05, 31.41, 23.17, 21.08, 20.5, 14.03, 15.33, 22.97, 17.43, 15.87, 13.13, 18.59, 13.3, 34.28, 23.77, 16.07, 21.88, 14.11, 33.13, 12.43, 15.8, 13.91, 14.8, 38.09, 20.41, 9.55, 23.66, 18.28, 36.82, 24.21, 15.36, 13.78, 20.0, 27.04, 18.88, 15.74, 16.62, 30.82, 30.54, 11.16, 20.77, 19.32, 19.56, 25.93, 15.39, 28.44, 21.26, 14.98, 22.7, 15.57, 25.26, 14.51, 18.12, 18.74, 12.77, 22.78, 34.38, 11.51, 17.97, 9.69, 15.31, 19.61, 16.34, 24.98, 26.57, 20.51, 9.94, 26.19]

def get_input_data500():
    return stocks, ESG

# stocks = []
# ESG = []

# for stock_symbol in snp500_stocks:
#     historical_data = yf.download(stock_symbol).reset_index()
#     time.sleep(1)
    
#     if historical_data.empty:
#         continue
    
#     historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.date
#     min_date = historical_data['Date'].min()
    
#     start_date = pd.to_datetime(start_date).date()
#     end_date = pd.to_datetime(end_date).date()
    
#     if min_date <= start_date:
#         stock_data = yf.Ticker(stock_symbol)
#         try:
#             if stock_data.sustainability is not None:
#                 ESG.append(stock_data.sustainability.loc['totalEsg'][0])
                
#                 stocks.append(stock_symbol)
                
#                 historical_data = historical_data[historical_data['Date'] >= start_date].reset_index(drop = True)
#                 historical_data = historical_data[historical_data['Date'] < end_date].reset_index(drop = True)
            
#                 date = historical_data["Date"]
#                 close = historical_data[["Close"]].squeeze()
#                 data = pd.DataFrame({"Date": date, "Close": close})
            
#                 csv_file_path = os.path.join(data_path, "DATA_" + stock_symbol + ".csv")
#                 data.to_csv(csv_file_path, index=False)
#         except:
#             pass

# print(list(stocks))
# print(list(ESG))
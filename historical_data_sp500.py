import yfinance as yf
import pandas as pd
import time
import os
import requests
import json

stocks= ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "AVGO", "LLY", "WMT", "JPM", "V", "MA", "XOM", "COST", "UNH", "ORCL", "PG", "JNJ", "NFLX", "HD", "BAC", "KO", "TMUS", "CRM", "CVX", "CSCO", "WFC", "PM", "MRK", "ABT", "IBM", "MCD", "LIN", "ACN", "PEP", "GE", "TMO", "AXP", "ADBE", "MS", "ISRG", "T", "DIS", "VZ", "GS", "BX", "QCOM", "TXN", "RTX", "AMGN", "INTU", "PGR", "CAT", "AMD", "SPGI", "BKNG", "DHR", "BLK", "BSX", "PFE", "UNP", "SYK", "GILD", "NEE", "SCHW", "HON", "CMCSA", "LOW", "TJX", "C", "DE", "VRTX", "FI", "AMAT", "ADP", "BMY", "SBUX", "MDT", "BA", "MMC", "NKE", "CB", "PLD", "COP", "ADI", "ETN", "LMT", "UPS", "MU", "LRCX", "ICE", "SO", "AMT", "MO", "WELL", "KLAC", "ELV", "CME", "WM", "SHW", "INTC", "DUK", "AON", "AJG", "CI", "MDLZ", "EQIX", "MCO", "CVS", "PH", "CTAS", "MCK", "MMM", "FTNT", "ITW", "TT", "ORLY", "CL", "TDG", "ECL", "APH", "REGN", "MAR", "RSG", "GD", "PNC", "MSI", "CMG", "APD", "USB", "SNPS", "NOC", "EOG", "COF", "CDNS", "WMB", "SPG", "EMR", "BDX", "ROP", "BK", "AZO", "FDX", "AFL", "CSX", "TRV", "RCL", "PCAR", "MET", "ADSK", "TFC", "AEP", "PSA", "MNST", "FCX", "SLB", "PAYX", "NSC", "OKE", "JCI", "ALL", "TGT", "CPRT", "O", "DLR", "NEM", "COR", "AMP", "AIG", "GWW", "CMI", "KMB", "KR", "D", "ROST", "KDP", "SRE", "FAST", "YUM", "FICO", "HES", "MSCI", "TEL", "NDAQ", "DFS", "EXC", "OXY", "GRMN", "AME", "LULU", "DHI", "EW", "BKR", "VRSK", "CCI", "CTSH", "URI", "LHX", "VLO", "CBRE", "PRU", "PEG", "GLW", "XEL", "AXON", "F", "ODFL", "FIS", "IT", "SYY", "TTWO", "HSY", "A", "PWR", "EA", "PCG", "DAL", "ED", "ETR", "IDXX", "GIS", "EXR", "ACGL", "BRO", "LEN", "HIG", "RMD", "WEC", "WTW", "DD", "LVS", "STZ", "HUM", "EBAY", "CSGP", "MCHP", "DXCM", "VMC", "AVB", "ROK", "CAH", "EFX", "NUE", "WAB", "LYV", "TPL", "VTR", "CNC", "RJF", "MTB", "MLM", "TSCO", "ANSS", "K", "HPQ", "UAL", "EQR", "BR", "EQT", "CCL", "CHD", "IP", "MPWR", "DTE", "AWK", "MTD", "FITB", "WBD", "AEE", "PPG", "TYL", "STT", "EL", "DOV", "ROL", "PPL", "GPN", "IRM", "EXPE", "WRB", "SBAC", "ERIE", "ATO", "ADM", "TDY", "WAT", "STE", "VRSN", "DRI", "NVR", "FE", "CINF", "TROW", "HBAN", "SMCI", "DVN", "MKC", "WY", "PHM", "CNP", "BIIB", "TSN", "ES", "CMS", "LH", "HAL", "EIX", "IFF", "ZBH", "LII", "NTRS", "ESS", "DECK", "MAA", "RF", "PFG", "CTRA", "PTC", "DGX", "STLD", "NTAP", "HUBB", "ON", "STX", "CLX", "PODD", "PKG", "COO", "NI", "BAX", "L", "NRG", "MOH", "KEY", "SNA", "LUV", "GPC", "LDOS", "ARE", "TER", "GEN", "WST", "BBY", "FDS", "DG", "EXPD", "TRMB", "DPZ", "ULTA", "OMC", "JBHT", "TPR", "UDR", "LNT", "HRL", "FFIV", "MAS", "EG", "BLDR", "J", "JBL", "ZBRA", "EVRG", "DLTR", "PNR", "BALL", "RL", "KIM", "WDC", "AVY", "IEX", "FSLR", "HOLX", "RVTY", "INCY", "REG", "POOL", "JKHY", "TXT", "CF", "ALGN", "AKAM", "CAG", "NDSN", "TAP", "JNPR", "KMX", "CPB", "SJM", "BXP", "CHRW", "UHS", "VTRS", "DVA", "HST", "EMN", "LKQ", "PNW", "SWKS", "BEN", "GL", "AIZ", "IPG", "TECH", "BG", "AOS", "WYNN", "WBA", "HSIC", "MGM", "ALB", "HAS", "FRT", "CRL", "PARA", "MTCH", "MOS", "MKTX", "AES", "MHK", "IVZ", "APA", "BWA", "TFX", "CE", "FMC"]
ESG_dict = {'AAPL': 16.36, 'MSFT': 13.51, 'NVDA': 12.23, 'AMZN': 26.1, 'GOOGL': 24.89, 'AVGO': 19.2, 'LLY': 23.62, 'WMT': 25.26, 'JPM': 27.3, 'V': 15.39, 'MA': 16.13, 'XOM': 43.66, 'COST': 29.12, 'UNH': 16.62, 'ORCL': 14.89, 'PG': 24.93, 'JNJ': 20.1, 'NFLX': 15.6, 'HD': 12.61, 'BAC': 24.36, 'KO': 24.16, 'TMUS': 22.97, 'CRM': 15.18, 'CVX': 38.36, 'CSCO': 12.91, 'WFC': 34.38, 'PM': 26.55, 'MRK': 19.96, 'ABT': 21.62, 'IBM': 13.27, 'MCD': 25.56, 'LIN': 11.63, 'ACN': 11.09, 'PEP': 19.98, 'GE': 32.49, 'TMO': 12.43, 'AXP': 18.28, 'ADBE': 14.06, 'MS': 24.82, 'ISRG': 18.03, 'T': 22.07, 'DIS': 14.51, 'VZ': 19.32, 'GS': 25.22, 'BX': 25.17, 'QCOM': 13.62, 'TXN': 21.88, 'RTX': 29.84, 'AMGN': 22.76, 'INTU': 16.82, 'PGR': 19.83, 'CAT': 28.33, 'AMD': 12.51, 'SPGI': 11.45, 'BKNG': 14.76, 'DHR': 8.52, 'BLK': 23.32, 'BSX': 22.06, 'PFE': 19.36, 'UNP': 20.0, 'SYK': 21.08, 'GILD': 21.68, 'NEE': 25.05, 'SCHW': 23.3, 'HON': 27.12, 'CMCSA': 22.27, 'LOW': 11.93, 'TJX': 15.8, 'C': 21.89, 'DE': 16.2, 'VRTX': 19.56, 'FI': 19.21, 'AMAT': 11.56, 'ADP': 15.12, 'BMY': 21.2, 'SBUX': 22.34, 'MDT': 22.16, 'BA': 36.55, 'MMC': 19.98, 'NKE': 18.44, 'CB': 23.3, 'PLD': 10.93, 'COP': 33.14, 'ADI': 16.7, 'ETN': 21.05, 'LMT': 28.14, 'UPS': 18.88, 'MU': 18.6, 'LRCX': 12.21, 'ICE': 18.24, 'SO': 28.14, 'AMT': 12.62, 'MO': 27.88, 'WELL': 11.51, 'KLAC': 13.08, 'ELV': 9.31, 'CME': 17.08, 'WM': 18.74, 'SHW': 29.38, 'INTC': 19.17, 'DUK': 27.16, 'AON': 15.29, 'AJG': 20.26, 'CI': 12.99, 'MDLZ': 21.98, 'EQIX': 13.14, 'MCO': 14.62, 'CVS': 18.27, 'PH': 26.84, 'CTAS': 17.03, 'MCK': 13.06, 'MMM': 42.86, 'FTNT': 15.96, 'ITW': 26.26, 'TT': 14.8, 'ORLY': 11.78, 'CL': 24.12, 'TDG': 38.09, 'ECL': 23.86, 'APH': 18.02, 'REGN': 18.39, 'MAR': 19.32, 'RSG': 17.04, 'GD': 34.09, 'PNC': 23.71, 'MSI': 17.87, 'CMG': 22.6, 'APD': 14.83, 'USB': 24.21, 'SNPS': 14.03, 'NOC': 26.61, 'EOG': 34.38, 'COF': 20.42, 'CDNS': 14.49, 'WMB': 19.61, 'SPG': 12.49, 'EMR': 22.79, 'BDX': 23.67, 'ROP': 19.42, 'BK': 19.93, 'AZO': 10.98, 'FDX': 19.14, 'AFL': 18.58, 'CSX': 21.07, 'TRV': 20.41, 'RCL': 18.45, 'PCAR': 31.38, 'MET': 15.09, 'ADSK': 15.14, 'TFC': 23.66, 'AEP': 21.95, 'PSA': 13.08, 'MNST': 32.89, 'FCX': 28.1, 'SLB': 18.48, 'PAYX': 16.67, 'NSC': 23.04, 'OKE': 26.37, 'JCI': 15.92, 'ALL': 24.68, 'TGT': 18.59, 'CPRT': 18.25, 'O': 15.53, 'DLR': 12.63, 'NEM': 21.39, 'COR': 12.43, 'AMP': 18.71, 'AIG': 23.6, 'GWW': 14.98, 'CMI': 22.82, 'KMB': 24.45, 'KR': 22.8, 'D': 28.58, 'ROST': 17.03, 'KDP': 24.29, 'SRE': 23.22, 'FAST': 25.04, 'YUM': 20.51, 'FICO': 20.09, 'HES': 31.97, 'MSCI': 14.4, 'TEL': 13.3, 'NDAQ': 13.63, 'DFS': 22.81, 'EXC': 18.98, 'OXY': 37.83, 'GRMN': 20.99, 'AME': 14.23, 'LULU': 14.11, 'DHI': 21.64, 'EW': 22.02, 'BKR': 19.12, 'VRSK': 16.29, 'CCI': 11.98, 'CTSH': 15.34, 'URI': 15.74, 'LHX': 24.2, 'VLO': 30.54, 'CBRE': 8.96, 'PRU': 18.63, 'PEG': 25.34, 'GLW': 17.5, 'XEL': 26.57, 'AXON': 30.49, 'F': 27.57, 'ODFL': 15.93, 'FIS': 18.24, 'IT': 16.67, 'SYY': 15.33, 'TTWO': 15.87, 'HSY': 24.98, 'A': 11.31, 'PWR': 36.6, 'EA': 12.96, 'PCG': 30.51, 'DAL': 30.38, 'ED': 21.11, 'ETR': 25.02, 'IDXX': 16.32, 'GIS': 26.0, 'EXR': 13.67, 'ACGL': 20.27, 'BRO': 21.11, 'LEN': 26.02, 'HIG': 16.78, 'RMD': 24.67, 'WEC': 22.78, 'WTW': 16.34, 'DD': 26.87, 'LVS': 20.65, 'STZ': 26.39, 'HUM': 18.9, 'EBAY': 15.2, 'CSGP': 21.68, 'MCHP': 29.56, 'DXCM': 20.24, 'VMC': 28.44, 'AVB': 8.09, 'ROK': 20.36, 'CAH': 11.32, 'EFX': 22.21, 'NUE': 31.93, 'WAB': 22.7, 'LYV': 17.24, 'TPL': 14.11, 'VTR': 11.16, 'CNC': 15.29, 'RJF': 26.68, 'MTB': 25.26, 'MLM': 23.9, 'TSCO': 13.91, 'ANSS': 14.69, 'K': 30.44, 'HPQ': 10.59, 'UAL': 27.04, 'EQR': 11.43, 'BR': 18.18, 'EQT': 31.69, 'CCL': 21.94, 'CHD': 22.65, 'IP': 22.22, 'MPWR': 19.03, 'DTE': 31.28, 'AWK': 19.18, 'MTD': 10.73, 'FITB': 16.87, 'WBD': 18.12, 'AEE': 26.97, 'PPG': 22.3, 'TYL': 18.28, 'STT': 22.05, 'EL': 23.96, 'DOV': 24.45, 'ROL': 18.6, 'PPL': 26.91, 'GPN': 19.63, 'IRM': 14.25, 'EXPE': 21.4, 'WRB': 21.26, 'SBAC': 9.73, 'ERIE': 27.91, 'ATO': 31.23, 'ADM': 33.7, 'TDY': 34.28, 'WAT': 12.77, 'STE': 23.17, 'VRSN': 20.77, 'DRI': 27.46, 'NVR': 21.0, 'FE': 27.51, 'CINF': 23.11, 'TROW': 17.43, 'HBAN': 16.38, 'SMCI': 20.5, 'DVN': 31.55, 'MKC': 26.04, 'WY': 15.31, 'PHM': 21.13, 'CNP': 24.96, 'BIIB': 20.51, 'TSN': 36.82, 'ES': 18.08, 'CMS': 20.26, 'LH': 19.74, 'HAL': 23.84, 'EIX': 23.97, 'IFF': 23.17, 'ZBH': 26.19, 'LII': 22.38, 'NTRS': 24.92, 'ESS': 11.61, 'DECK': 14.23, 'MAA': 11.68, 'RF': 14.3, 'PFG': 11.18, 'CTRA': 32.75, 'PTC': 18.48, 'DGX': 20.49, 'STLD': 31.41, 'NTAP': 14.34, 'HUBB': 17.94, 'ON': 21.27, 'STX': 11.55, 'CLX': 20.03, 'PODD': 21.37, 'PKG': 17.64, 'COO': 14.69, 'NI': 20.58, 'BAX': 21.91, 'L': 28.56, 'NRG': 34.25, 'MOH': 22.84, 'KEY': 25.21, 'SNA': 30.88, 'LUV': 28.41, 'GPC': 12.83, 'LDOS': 17.26, 'ARE': 13.14, 'TER': 16.07, 'GEN': 14.61, 'WST': 17.97, 'BBY': 13.67, 'FDS': 15.51, 'DG': 21.2, 'EXPD': 18.4, 'TRMB': 9.55, 'DPZ': 27.49, 'ULTA': 13.78, 'OMC': 14.57, 'JBHT': 14.34, 'TPR': 13.13, 'UDR': 15.36, 'LNT': 17.11, 'HRL': 26.11, 'FFIV': 16.3, 'MAS': 22.27, 'EG': 17.96, 'BLDR': 26.73, 'J': 23.89, 'JBL': 9.45, 'ZBRA': 9.94, 'EVRG': 29.24, 'DLTR': 18.8, 'PNR': 22.0, 'BALL': 13.06, 'RL': 13.91, 'KIM': 11.22, 'WDC': 9.69, 'AVY': 10.98, 'IEX': 30.65, 'FSLR': 17.31, 'HOLX': 21.85, 'RVTY': 15.58, 'INCY': 23.59, 'REG': 11.31, 'POOL': 10.93, 'JKHY': 18.66, 'TXT': 33.13, 'CF': 28.16, 'ALGN': 18.58, 'AKAM': 13.31, 'CAG': 26.94, 'NDSN': 24.15, 'TAP': 27.98, 'JNPR': 13.28, 'KMX': 11.08, 'CPB': 26.3, 'SJM': 27.19, 'BXP': 11.99, 'CHRW': 17.44, 'UHS': 30.82, 'VTRS': 25.93, 'DVA': 21.78, 'HST': 12.86, 'EMN': 34.47, 'LKQ': 10.99, 'PNW': 25.96, 'SWKS': 24.11, 'BEN': 19.67, 'GL': 20.43, 'AIZ': 23.48, 'IPG': 8.89, 'TECH': 25.97, 'BG': 32.92, 'AOS': 26.82, 'WYNN': 24.98, 'WBA': 15.57, 'HSIC': 14.5, 'MGM': 24.27, 'ALB': 20.11, 'HAS': 7.28, 'FRT': 12.44, 'CRL': 18.14, 'PARA': 13.64, 'MTCH': 16.89, 'MOS': 37.28, 'MKTX': 13.87, 'AES': 24.32, 'MHK': 13.74, 'IVZ': 21.61, 'APA': 43.08, 'BWA': 9.93, 'TFX': 23.77, 'CE': 24.96, 'FMC': 25.08}
ESG = [16.36, 13.51, 12.23, 26.1, 24.89, 19.2, 23.62, 25.26, 27.3, 15.39, 16.13, 43.66, 29.12, 16.62, 14.89, 24.93, 20.1, 15.6, 12.61, 24.36, 24.16, 22.97, 15.18, 38.36, 12.91, 34.38, 26.55, 19.96, 21.62, 13.27, 25.56, 11.63, 11.09, 19.98, 32.49, 12.43, 18.28, 14.06, 24.82, 18.03, 22.07, 14.51, 19.32, 25.22, 25.17, 13.62, 21.88, 29.84, 22.76, 16.82, 19.83, 28.33, 12.51, 11.45, 14.76, 8.52, 23.32, 22.06, 19.36, 20.0, 21.08, 21.68, 25.05, 23.3, 27.12, 22.27, 11.93, 15.8, 21.89, 16.2, 19.56, 19.21, 11.56, 15.12, 21.2, 22.34, 22.16, 36.55, 19.98, 18.44, 23.3, 10.93, 33.14, 16.7, 21.05, 28.14, 18.88, 18.6, 12.21, 18.24, 28.14, 12.62, 27.88, 11.51, 13.08, 9.31, 17.08, 18.74, 29.38, 19.17, 27.16, 15.29, 20.26, 12.99, 21.98, 13.14, 14.62, 18.27, 26.84, 17.03, 13.06, 42.86, 15.96, 26.26, 14.8, 11.78, 24.12, 38.09, 23.86, 18.02, 18.39, 19.32, 17.04, 34.09, 23.71, 17.87, 22.6, 14.83, 24.21, 14.03, 26.61, 34.38, 20.42, 14.49, 19.61, 12.49, 22.79, 23.67, 19.42, 19.93, 10.98, 19.14, 18.58, 21.07, 20.41, 18.45, 31.38, 15.09, 15.14, 23.66, 21.95, 13.08, 32.89, 28.1, 18.48, 16.67, 23.04, 26.37, 15.92, 24.68, 18.59, 18.25, 15.53, 12.63, 21.39, 12.43, 18.71, 23.6, 14.98, 22.82, 24.45, 22.8, 28.58, 17.03, 24.29, 23.22, 25.04, 20.51, 20.09, 31.97, 14.4, 13.3, 13.63, 22.81, 18.98, 37.83, 20.99, 14.23, 14.11, 21.64, 22.02, 19.12, 16.29, 11.98, 15.34, 15.74, 24.2, 30.54, 8.96, 18.63, 25.34, 17.5, 26.57, 30.49, 27.57, 15.93, 18.24, 16.67, 15.33, 15.87, 24.98, 11.31, 36.6, 12.96, 30.51, 30.38, 21.11, 25.02, 16.32, 26.0, 13.67, 20.27, 21.11, 26.02, 16.78, 24.67, 22.78, 16.34, 26.87, 20.65, 26.39, 18.9, 15.2, 21.68, 29.56, 20.24, 28.44, 8.09, 20.36, 11.32, 22.21, 31.93, 22.7, 17.24, 14.11, 11.16, 15.29, 26.68, 25.26, 23.9, 13.91, 14.69, 30.44, 10.59, 27.04, 11.43, 18.18, 31.69, 21.94, 22.65, 22.22, 19.03, 31.28, 19.18, 10.73, 16.87, 18.12, 26.97, 22.3, 18.28, 22.05, 23.96, 24.45, 18.6, 26.91, 19.63, 14.25, 21.4, 21.26, 9.73, 27.91, 31.23, 33.7, 34.28, 12.77, 23.17, 20.77, 27.46, 21.0, 27.51, 23.11, 17.43, 16.38, 20.5, 31.55, 26.04, 15.31, 21.13, 24.96, 20.51, 36.82, 18.08, 20.26, 19.74, 23.84, 23.97, 23.17, 26.19, 22.38, 24.92, 11.61, 14.23, 11.68, 14.3, 11.18, 32.75, 18.48, 20.49, 31.41, 14.34, 17.94, 21.27, 11.55, 20.03, 21.37, 17.64, 14.69, 20.58, 21.91, 28.56, 34.25, 22.84, 25.21, 30.88, 28.41, 12.83, 17.26, 13.14, 16.07, 14.61, 17.97, 13.67, 15.51, 21.2, 18.4, 9.55, 27.49, 13.78, 14.57, 14.34, 13.13, 15.36, 17.11, 26.11, 16.3, 22.27, 17.96, 26.73, 23.89, 9.45, 9.94, 29.24, 18.8, 22.0, 13.06, 13.91, 11.22, 9.69, 10.98, 30.65, 17.31, 21.85, 15.58, 23.59, 11.31, 10.93, 18.66, 33.13, 28.16, 18.58, 13.31, 26.94, 24.15, 27.98, 13.28, 11.08, 26.3, 27.19, 11.99, 17.44, 30.82, 25.93, 21.78, 12.86, 34.47, 10.99, 25.96, 24.11, 19.67, 20.43, 23.48, 8.89, 25.97, 32.92, 26.82, 24.98, 15.57, 14.5, 24.27, 20.11, 7.28, 12.44, 18.14, 13.64, 16.89, 37.28, 13.87, 24.32, 13.74, 21.61, 43.08, 9.93, 23.77, 24.96, 25.08]

def get_input_data500():
    return stocks, ESG


# ESG_array = ESG_dict.values()

# print(ESG_array)

# esg_data = {}
# nodata = []

# for ticker in stocks:
#     data = yf.Ticker(ticker)
#     try:
#         if data.sustainability is not None:
#             temp = pd.DataFrame.transpose(data.sustainability)
#             temp['company_ticker'] = str(data.ticker)
#             esg_score = temp['totalEsg'].esgScores
#             esg_data[ticker] = esg_score
#     except KeyError as e:
#         print(e)
#         nodata.append(ticker)
#         pass

# print(esg_data)

# print(nodata)


# Define time period
# start_date = "2009-01-01"
# end_date = "2025-01-01"

# data_path = "./DATA"
# os.makedirs(data_path, exist_ok=True)

# no_data_tickers = []

# def check_years_covered(data, start_year, end_year):
#     available_years = set(pd.to_datetime(data["Date"]).dt.year)
#     required_years = set(range(start_year, end_year + 1))
#     return required_years.issubset(available_years)

# for stock_symbol in stocks:
#     historical_data = yf.download(stock_symbol).reset_index()
#     time.sleep(1)
#     historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.date
    
#     min_date = pd.to_datetime(start_date).date()
#     max_date = pd.to_datetime(end_date).date()
    
#     historical_data = historical_data[(historical_data['Date'] >= min_date) & (historical_data['Date'] < max_date)]
    
#     if not check_years_covered(historical_data, 2009, 2024):
#         print(f"Missing data for {stock_symbol}")
#         no_data_tickers.append(stock_symbol)
    
#     csv_file_path = os.path.join(data_path, f"DATA_{stock_symbol}.csv")
#     historical_data.to_csv(csv_file_path, index=False)

# print("Stocks with missing data:", no_data_tickers)

# Hierarchical Clustering of Stocks Based on Returns and ESG Scores

## Overview

This project applies hierarchical clustering to stocks based on historical returns and ESG scores. The goal is to form stock clusters that optimize portfolio returns while satisfying the ESG constraints.

## Dependencies

This project relies on several Python libraries for data handling, visualization, and clustering. Make sure the following packages are installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `plotly`
- `yfinance`
- `kneed`

## File Descriptions

- **`download_historical_stock_data.py`**: Retrieves historical stock prices and ESG scores using Yahoo Finance.
- **`cluster_number.py`**: Provides different methods to determine the optimal number of clusters and visualizes the results.
- **`Hierarchical Clustering of Returns and ESG.py`**: The mainscript loads data, computes metrics, applies hierarchical clustering, optimizes the portfolio, and provides various visualizations of the results.

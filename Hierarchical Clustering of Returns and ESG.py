import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from sklearn.preprocessing import MinMaxScaler

# List of Dow Jones stocks
stocks = ["AAPL", "NVDA", "MSFT", "AMZN", "WMT", "JPM", "V", "UNH", "PG", 
          "JNJ", "HD", "KO", "CRM", "CVX", "CSCO", "IBM", "MRK", "MCD", "AXP", 
          "GS", "DIS", "VZ", "CAT", "AMGN", "HON", "BA", "NKE", "SHW", "MMM", "TRV"]

# ESG Risk Ratings from Sustainalytics
ESG = [18.7, 12.2, 13.7, 26.1, 25.3, 27.3, 15.4, 16.6, 24.9, 19.9, 12.6, 24.2,
       18.1, 38.4, 12.9, 13.3, 20.2, 25.6, 18.3, 25.2, 15.9, 19.3, 28.3, 22.8, 
       26.2, 36.6, 18.4, 29.4, 42.9, 20.4]

# Normalize ESG scores to [0, 1]
scaler = MinMaxScaler()
ESG_normalized = scaler.fit_transform(np.array(ESG).reshape(-1, 1)).flatten()

annual_returns_dict = {}

# Read stock data and compute annual returns
for stock_symbol in stocks:
    file_path = f"DATA/DATA_{stock_symbol}.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    stock_data = pd.read_csv(file_path)
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data = stock_data.sort_values("Date")
    stock_data["Year"] = stock_data["Date"].dt.year
    annual_returns = stock_data.groupby("Year")["Close"].apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
    annual_returns_dict[stock_symbol] = annual_returns

# Convert annual returns dictionary into a dataframe
returns_df = pd.DataFrame(annual_returns_dict)

# Compute correlation matrix and distance matrix
# Chosen distance measure is squared distance to emphasize larger differences
corr_matrix = returns_df.corr()
corr_dist_matrix = np.square(1 - corr_matrix)

# Compute ESG distance matrix
esg_dist_matrix = np.abs(ESG_normalized[:, None] - ESG_normalized[None, :])
esg_dist_matrix = pd.DataFrame(esg_dist_matrix, index=stocks, columns=stocks)

# Set parameter alpha (0 means only correlation, 1 means only ESG)
alpha = 0.75

# Compute the combined correlation and ESG distance matrix
combined_dist_matrix = (1 - alpha) * corr_dist_matrix + alpha * esg_dist_matrix

# Perform hierarchical clustering
linkage_method = "average"  # Choose "single", "complete", "average", "ward", etc.
linkage_matrix = linkage(combined_dist_matrix, method=linkage_method)

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=combined_dist_matrix.index, leaf_rotation=90)
plt.title("Dendrogram for alpha = " + str(alpha) + " (Note: 0 is only correlation, 1 is only ESG)")
plt.xlabel("Stocks")
plt.ylabel("Distance")
plt.show()

# Reorder based on clustering
reordered_indices = leaves_list(linkage_matrix)
reordered_corr_dist_matrix = corr_dist_matrix.iloc[reordered_indices, reordered_indices]
reordered_esg_dist_matrix = esg_dist_matrix.iloc[reordered_indices, reordered_indices]
combined_dist_matrix_reordered = combined_dist_matrix.iloc[reordered_indices, reordered_indices]

# Plot original and reordered matrices
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

sns.heatmap(corr_dist_matrix, ax=axes[0, 0], cmap="coolwarm", annot=False)
axes[0, 0].set_title("Original Correlation Distance Matrix")

sns.heatmap(esg_dist_matrix, ax=axes[0, 1], cmap="viridis", annot=False)
axes[0, 1].set_title("Original ESG Distance Matrix")

sns.heatmap(combined_dist_matrix, ax=axes[0, 2], cmap="cividis", annot=False)
axes[0, 2].set_title("Original Combined Distance Matrix")

sns.heatmap(reordered_corr_dist_matrix, ax=axes[1, 0], cmap="coolwarm", annot=False)
axes[1, 0].set_title("Reordered Correlation Distance Matrix")

sns.heatmap(reordered_esg_dist_matrix, ax=axes[1, 1], cmap="viridis", annot=False)
axes[1, 1].set_title("Reordered ESG Distance Matrix")

sns.heatmap(combined_dist_matrix_reordered, ax=axes[1, 2], cmap="cividis", annot=False)
axes[1, 2].set_title("Reordered Combined Distance Matrix")

plt.tight_layout()
plt.show()



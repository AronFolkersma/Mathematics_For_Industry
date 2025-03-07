import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from cluster_number import plot_optimal_number_of_clusters
from historical_stock_data import get_input_data30
from historical_data_sp500 import get_input_data500
import matplotlib.patches as patches

stocks, ESG = get_input_data500()

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
    annual_returns = stock_data.groupby("Year")["Close"].apply(lambda x: (float(x.iloc[-1]) - float(x.iloc[0])) / float(x.iloc[0]))
    annual_returns_dict[stock_symbol] = annual_returns

# Convert annual returns dictionary into a dataframe
returns_df = pd.DataFrame(annual_returns_dict)

# Compute correlation matrix and distance matrix
corr_matrix = returns_df.corr()
corr_dist_matrix = np.sqrt(0.5*(1 - corr_matrix))

# Compute ESG distance matrix
esg_dist_matrix = np.abs(ESG_normalized[:, None] - ESG_normalized[None, :])
esg_dist_matrix = pd.DataFrame(esg_dist_matrix, index=stocks, columns=stocks)

# Set parameter alpha (0 means only correlation, 1 means only ESG)
alpha = 0.15

# Compute the combined correlation and ESG distance matrix
combined_dist_matrix = (1 - alpha) * corr_dist_matrix + alpha * esg_dist_matrix

# Perform hierarchical clustering
linkage_method = "ward"  # Choose "single", "complete", "average", "ward", etc.
linkage_matrix = linkage(squareform(combined_dist_matrix), method=linkage_method, optimal_ordering=True)

# Extract cluster assignments
num_clusters = 3  # Adjust as needed
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Create a DataFrame for cluster assignments
cluster_df = pd.DataFrame({'Stock': combined_dist_matrix.index, 'Cluster': clusters})
cluster_df = cluster_df.sort_values(by="Cluster").reset_index(drop=True)

# Compute average ESG score per cluster 
esg_df = pd.DataFrame({'Stock': stocks, 'ESG': ESG, 'Cluster': clusters})
avg_esg_per_cluster = esg_df.groupby("Cluster")["ESG"].mean()

# Compute average return per year per cluster
returns_cluster = returns_df.stack().reset_index()
returns_cluster.columns = ["Year", "Stock", "Return"]
returns_cluster = returns_cluster.merge(cluster_df, on="Stock")
avg_returns_per_cluster_per_year = returns_cluster.groupby(["Year", "Cluster"])["Return"].mean().reset_index()

# Compute the covariance matrix, correlation matrix between and total average return of cluster
avg_cluster_returns_wide = avg_returns_per_cluster_per_year.pivot(index="Year", columns="Cluster", values="Return")
cluster_cov_matrix = avg_cluster_returns_wide.cov()
cluster_corr_matrix = avg_cluster_returns_wide.corr()
avg_return_per_cluster = returns_cluster.groupby("Cluster")["Return"].mean().reset_index()

# Check invertibility of covariance matrix
condition_number = np.linalg.cond(cluster_cov_matrix)
print(f"Condition number of covariance matrix: {condition_number:.2f}")

# Print cluster groupings
for cluster_num in range(1, num_clusters + 1):
    cluster_members = cluster_df[cluster_df['Cluster'] == cluster_num]['Stock'].tolist()
    print(
    f"Cluster {cluster_num} (ESG: {avg_esg_per_cluster[cluster_num]:.2f}) "
    f"(Expected return: {avg_return_per_cluster.loc[cluster_num - 1, 'Return']:.2f}): "
    f"{', '.join(cluster_members)}"
)


# # Plot dendrogram without clusters
# plt.figure(figsize=(12, 6))
# dendrogram_output = dendrogram(linkage_matrix, labels=combined_dist_matrix.index, leaf_rotation=90)
# plt.title(f"Dendrogram for alpha = {alpha} with linkage method: " + linkage_method + 
#           " (Note: 0 is only correlation, 1 is only ESG)")
# plt.ylabel("Distance")
# plt.show()

# Counter to track function calls
label_counter = 0

# Define a function to label the leaves with cluster numbers
def leaf_label_func(id):
    global label_counter
    label_counter += 1
    return f"Cluster {label_counter}"


# Find Optimal Number of Clusters
methods = ["maxgap", "elbow", "average silhouette"]
plot_optimal_number_of_clusters(linkage_matrix, combined_dist_matrix, len(stocks), methods, stocks)

# Plot dendrogram with clusters
plt.figure(figsize=(12, 6))
dendrogram_output = dendrogram(linkage_matrix, p=num_clusters, truncate_mode='lastp', 
                               leaf_rotation=45, leaf_label_func=leaf_label_func)
plt.title(f"Dendrogram for alpha = {alpha} with linkage method: " + linkage_method + 
          " (Note: 0 is only correlation, 1 is only ESG)")
plt.ylabel("Distance")
plt.show()

# Reorder based on clustering
reordered_indices = leaves_list(linkage_matrix)
reordered_corr_dist_matrix = corr_dist_matrix.iloc[reordered_indices, reordered_indices]
reordered_esg_dist_matrix = esg_dist_matrix.iloc[reordered_indices, reordered_indices]
combined_dist_matrix_reordered = combined_dist_matrix.iloc[reordered_indices, reordered_indices]
reordered_clusters = clusters[reordered_indices]

# Find cluster boundaries
boundaries = np.where(np.diff(reordered_clusters) != 0)[0] + 1

# Function to plot heatmap with cluster borders
def plot_heatmap_with_borders(matrix, title, ax, cmap):
    sns.heatmap(matrix, ax=ax, cmap=cmap, annot=False)

    # Add white borders around clusters
    for i in range(len(boundaries) + 1):
        if i == 0:
            start = 0
        else:
            start = boundaries[i - 1]
        if i == len(boundaries):
            end = len(reordered_indices)
        else:
            end = boundaries[i]

        # Draw a white rectangle around each cluster
        rect = patches.Rectangle((start, start), end - start, end - start,
            linewidth=2, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
    
    ax.set_title(title)

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

# Plot reordered heatmaps with cluster borders
plot_heatmap_with_borders(reordered_corr_dist_matrix, "Reordered Correlation Distance Matrix", axes[1, 0], "coolwarm")
plot_heatmap_with_borders(reordered_esg_dist_matrix, "Reordered ESG Distance Matrix", axes[1, 1], "viridis")
plot_heatmap_with_borders(combined_dist_matrix_reordered, "Reordered Combined Distance Matrix", axes[1, 2], "cividis")


sns.heatmap(reordered_esg_dist_matrix, ax=axes[1, 1], cmap="viridis", annot=False)
axes[1, 1].set_title("Reordered ESG Distance Matrix")

sns.heatmap(combined_dist_matrix_reordered, ax=axes[1, 2], cmap="cividis", annot=False)
axes[1, 2].set_title("Reordered Combined Distance Matrix")

plt.tight_layout()
plt.show()

# Optimzation problem
def objective(weights, returns, cov_matrix, risk_aversion):
    return - (weights @ returns - (risk_aversion / 2) * (weights @ cov_matrix @ weights))

# Bounds: No shorting (long-only portfolio)
bounds = [(0, 1) for _ in range(len(avg_return_per_cluster))]

# Solve optimization
risk_aversion = 12  # Adjust risk aversion (low value is high return/high risk and high value low return/low risk)
lambda_constraint = 20  # Adjust ESG constraint (weighted ESG score must be lower than λ)
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested constraint
    {'type': 'ineq', 'fun': lambda w: lambda_constraint - w @ avg_esg_per_cluster.values}  # ESG should be below λ
]
init_weights = np.ones(len(avg_return_per_cluster)) / len(avg_return_per_cluster)
result = minimize(objective, init_weights, args=(avg_return_per_cluster['Return'].values, cluster_cov_matrix, risk_aversion),
                  constraints=constraints, bounds=bounds, method='SLSQP')

optimal_weights = result.x
np.set_printoptions(suppress=False, precision=2)
print("Optimal weights per cluster:\n", optimal_weights)

# Compute the portfolio ESG score (weighted average)
portfolio_esg_score = np.dot(optimal_weights, avg_esg_per_cluster.values)
print(f"Weighted ESG score of portfolio: {portfolio_esg_score:.2f}")

# Compute the portfolio expected return (weighted average)
portfolio_expected_return = np.dot(optimal_weights, avg_return_per_cluster["Return"].values)
print(f"Weighted expected return of portfolio: {portfolio_expected_return:.2f}")

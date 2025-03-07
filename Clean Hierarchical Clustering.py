import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as patches

# Counter to track function calls
label_counter = 0

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



def cor_metric(returns_df, p = 0):

    # Parameters
    # returns_df: dataframe of returns of the stocks
    # p: index which decides which correlation metric to use    
    

    # Calculate sample correlation from returns
    corr_matrix = returns_df.corr()

    # Choose which metric to use
    if p==0:
            return np.sqrt(0.5 * (1 - corr_matrix))

    if p==1:
            return None

    return None


def ESG_metric(ESG_normalized):
        
    #Parameters:
    # ESG_normalized: list of normalized ESG scores of the stocks


    # Calculate the difference between ESG scores and make it a matrix
    esg_dist_matrix = np.abs(ESG_normalized[:, None] - ESG_normalized[None, :])

    # return the matrix as a dataframe
    return  pd.DataFrame(esg_dist_matrix, index=stocks, columns=stocks)


def Cluster(alpha = 0, annual_returns_dict = annual_returns_dict, ESG_normalized = ESG_normalized, linkage_method = "ward", num_clusters = 6):

    # Parameters
    # alpha: constant that decides how important ESG is
    # annual_returns_dict: dictionary of historical returns
    # ESG: ESG scores of provided assets
    # linkage_method: choose "single", "complete", "average", "ward", etc.
    # num_clusters: number of clusters


    # Make dataframe of returns of stock
    returns_df = pd.DataFrame(annual_returns_dict)


    # Calculate both metrics
    global corr_dist_matrix; corr_dist_matrix = cor_metric(returns_df, p =0)
    global esg_dist_matrix; esg_dist_matrix = ESG_metric(ESG_normalized)

    # Combine them based on alpha
    global combined_dist_matrix; combined_dist_matrix = (1 - alpha) * corr_dist_matrix + alpha * esg_dist_matrix    

    # Create linkage matrix based on linkage:
    global linkage_matrix; linkage_matrix = linkage(squareform(combined_dist_matrix), method=linkage_method, optimal_ordering=True)



    # Create num_clusters amount of clusters:
    global clusters; clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    # Create a DataFrame for cluster assignments
    cluster_df = pd.DataFrame({'Stock': combined_dist_matrix.index, 'Cluster': clusters})
    cluster_df = cluster_df.sort_values(by="Cluster").reset_index(drop=True)


    # Compute average ESG score per cluster 
    esg_df = pd.DataFrame({'Stock': stocks, 'ESG': ESG, 'Cluster': clusters})
    global avg_esg_per_cluster; avg_esg_per_cluster = esg_df.groupby("Cluster")["ESG"].mean()

    # Create a dataframe for return of clusters instead of stocks themselves
    returns_cluster = (
    returns_df.stack()
    .reset_index(name="Return")
    .rename(columns={"level_0": "Year", "level_1": "Stock"})
    .merge(cluster_df, on="Stock")
    )

    # Calculate mean return of a cluster and convert to wide format (Years as index, Clusters as columns)
    avg_returns_per_cluster_per_year = returns_cluster.groupby(["Year", "Cluster"])["Return"].mean().reset_index()
    avg_cluster_returns_wide = avg_returns_per_cluster_per_year.pivot(index="Year", columns="Cluster", values="Return")


    # Compute the covariance matrix, correlation matrix between and total average return of cluster
    #! THIS IS DIFFERENT FROM DISCUSSED IN GROUP
    avg_cluster_returns_wide = avg_returns_per_cluster_per_year.pivot(index="Year", columns="Cluster", values="Return")
    global cluster_cov_matrix; cluster_cov_matrix = avg_cluster_returns_wide.cov()
    #cluster_corr_matrix = avg_cluster_returns_wide.corr()

    # Average return over entire timeframe
    global avg_return_per_cluster; avg_return_per_cluster = returns_cluster.groupby("Cluster")["Return"].mean().reset_index()



    # Print the resulting clusters (which stocks; average ESG; average return)
    for cluster_num in range(1, num_clusters + 1):
        cluster_members = cluster_df[cluster_df['Cluster'] == cluster_num]['Stock'].tolist()
        print(  f"Cluster {cluster_num} (ESG: {avg_esg_per_cluster[cluster_num]:.2f}) "
                f"(Expected return: {avg_return_per_cluster.loc[cluster_num - 1, 'Return']:.2f}): "
                f"{', '.join(cluster_members)}"
            )


    # Print condition number of cluster covariance matrix
    condition_number = np.linalg.cond(cluster_cov_matrix)
    print(f"Condition number of covariance matrix: {condition_number:.2f}")


    #! DONT REALLY KNOW WHAT THIS DOES BUT SURE
    # Plot dendrogram without clusters
    plt.figure(figsize=(12, 6))
    dendrogram_output = dendrogram(linkage_matrix, labels=combined_dist_matrix.index, leaf_rotation=90)
    plt.title(f"Dendrogram for alpha = {alpha} with linkage method: " + linkage_method + " (Note: 0 is only correlation, 1 is only ESG)")
    plt.ylabel("Distance")
    plt.show()

    # Plot dendrogram with clusters
    plt.figure(figsize=(12, 6))
    dendrogram_output = dendrogram(linkage_matrix, p=num_clusters, truncate_mode='lastp', 
                                leaf_rotation=45, leaf_label_func=leaf_label_func)
    plt.title(f"Dendrogram for alpha = {alpha} with linkage method: " + linkage_method + 
            " (Note: 0 is only correlation, 1 is only ESG)")
    plt.ylabel("Distance")
    plt.show()


    # The return probably has to change to make it more convenitent for other function calls.
    return None


# Define a function to label the leaves with cluster numbers
def leaf_label_func(id):
    global label_counter
    label_counter += 1
    return f"Cluster {label_counter}"




def plot_reordered_heatmaps(linkage_matrix, corr_dist_matrix, esg_dist_matrix, combined_dist_matrix, clusters):
    """
    Reorders the distance matrices based on hierarchical clustering, finds the cluster boundaries,
    and then plots the original and reordered heatmaps with white borders outlining the clusters.

    Parameters:
        linkage_matrix : ndarray
            Linkage matrix from hierarchical clustering.
        corr_dist_matrix : DataFrame
            Correlation distance matrix.
        esg_dist_matrix : DataFrame
            ESG distance matrix.
        combined_dist_matrix : DataFrame
            Combined distance matrix.
        clusters : array-like
            Cluster assignments corresponding to the distance matrices.
    """
    # Reorder indices based on the clustering
    reordered_indices = leaves_list(linkage_matrix)
    reordered_corr_dist_matrix = corr_dist_matrix.iloc[reordered_indices, reordered_indices]
    reordered_esg_dist_matrix = esg_dist_matrix.iloc[reordered_indices, reordered_indices]
    combined_dist_matrix_reordered = combined_dist_matrix.iloc[reordered_indices, reordered_indices]
    reordered_clusters = clusters[reordered_indices]
    
    # Find the boundaries where clusters change
    boundaries = np.where(np.diff(reordered_clusters) != 0)[0] + 1

    # Inner function to plot a heatmap with cluster borders
    def plot_heatmap_with_borders(matrix, title, ax, cmap):
        sns.heatmap(matrix, ax=ax, cmap=cmap, annot=False)
        # Add white borders around clusters
        for i in range(len(boundaries) + 1):
            start = 0 if i == 0 else boundaries[i - 1]
            end = len(reordered_indices) if i == len(boundaries) else boundaries[i]
            rect = patches.Rectangle((start, start), end - start, end - start,
                                     linewidth=2, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(title)

    # Set up the subplots: first row for original matrices, second row for reordered matrices
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # Plot original heatmaps
    sns.heatmap(corr_dist_matrix, ax=axes[0, 0], cmap="coolwarm", annot=False)
    axes[0, 0].set_title("Original Correlation Distance Matrix")
    
    sns.heatmap(esg_dist_matrix, ax=axes[0, 1], cmap="viridis", annot=False)
    axes[0, 1].set_title("Original ESG Distance Matrix")
    
    sns.heatmap(combined_dist_matrix, ax=axes[0, 2], cmap="cividis", annot=False)
    axes[0, 2].set_title("Original Combined Distance Matrix")

    # Plot reordered heatmaps with cluster borders
    plot_heatmap_with_borders(reordered_corr_dist_matrix, "Reordered Correlation Distance Matrix", axes[1, 0], "coolwarm")
    plot_heatmap_with_borders(reordered_esg_dist_matrix, "Reordered ESG Distance Matrix", axes[1, 1], "viridis")
    plot_heatmap_with_borders(combined_dist_matrix_reordered, "Reordered Combined Distance Matrix", axes[1, 2], "cividis")

    plt.tight_layout()
    plt.show()





# Solve optimization problem
def optimize_portfolio(avg_return_per_cluster, cluster_cov_matrix, avg_esg_per_cluster, risk_aversion=12, lambda_constraint=20):
    """

    Parameters
    ----------
    avg_return_per_cluster : pandas.DataFrame
        DataFrame containing the average returns for each cluster. It must include a column named 'Return'.
    cluster_cov_matrix : numpy.ndarray
        The covariance matrix corresponding to the clusters.
    avg_esg_per_cluster : pandas.Series or numpy.ndarray
        The ESG scores for each cluster. These scores will be used in the ESG constraint.
    risk_aversion : float, optional
        The risk aversion coefficient (default is 12). Lower values lead to higher return/higher risk, while higher values favor lower risk.
    lambda_constraint : float, optional
        The upper bound for the weighted ESG score (default is 20). The portfolio’s weighted ESG score must not exceed this threshold.

    Returns
    -------
    result : OptimizeResult
        The optimization result returned by scipy.optimize.minimize. The optimal weights can be accessed via result.x.
    """

    def objective(weights, returns, cov_matrix, risk_aversion):
        # Negative objective for maximization (we maximize return adjusted for risk)
        return - (weights @ returns - (risk_aversion / 2) * (weights @ cov_matrix @ weights))

    # Number of clusters/assets
    n = len(avg_return_per_cluster)

    # Define bounds (long-only portfolio: weights between 0 and 1)
    bounds = [(0, 1) for _ in range(n)]

    # Initial guess: equally weighted portfolio
    init_weights = np.ones(n) / n

    # Constraints:
    # 1. Fully invested constraint: sum of weights equals 1.
    # 2. ESG constraint: weighted ESG score must be below lambda_constraint.
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: lambda_constraint - w @ np.array(avg_esg_per_cluster)}
    ]

    # Run the optimization
    result = minimize(
        objective,
        init_weights,
        args=(avg_return_per_cluster['Return'].values, cluster_cov_matrix, risk_aversion),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )



    optimal_weights = result.x
    np.set_printoptions(suppress=False, precision=2)
    print("Optimal weights per cluster:\n", optimal_weights)

    # Compute the portfolio ESG score (weighted average)
    portfolio_esg_score = np.dot(optimal_weights, avg_esg_per_cluster.values)
    print(f"Weighted ESG score of portfolio: {portfolio_esg_score:.2f}")

    # Compute the portfolio expected return (weighted average)
    portfolio_expected_return = np.dot(optimal_weights, avg_return_per_cluster["Return"].values)
    print(f"Weighted expected return of portfolio: {portfolio_expected_return:.2f}")
    
    return result


Cluster(alpha = 1)
plot_reordered_heatmaps(linkage_matrix, corr_dist_matrix, esg_dist_matrix, combined_dist_matrix, clusters)
optimize_portfolio(avg_return_per_cluster, cluster_cov_matrix, avg_esg_per_cluster, risk_aversion=12, lambda_constraint=20)

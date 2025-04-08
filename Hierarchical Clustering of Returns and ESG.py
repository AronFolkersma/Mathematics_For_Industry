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
from download_historical_stock_data import get_input_data500
import plotly.graph_objects as go
import matplotlib.patches as patches

# Counter to track function calls for dendrogram
label_counter = 0

stocks, ESG = get_input_data500()

# Normalize ESG scores to [0, 1]
scaler = MinMaxScaler()
ESG_normalized = scaler.fit_transform(np.array(ESG).reshape(-1, 1)).flatten()

annual_returns_dict = {}

# Read stock data and compute annual returns
for stock_symbol in stocks:
    file_path = f"DATA/DATA/DATA_{stock_symbol}.csv"
    
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

def corr_metric(returns_df, p):
    # Compute correlation matrix and distance matrix
    corr_matrix = returns_df.corr()

    # Choose correlation metric
    if (p == 0):
            return np.sqrt(0.5*(1 - corr_matrix))
        
    else:
        d = np.sqrt(0.5 * (1 - corr_matrix)).values
        corr_dist_matrix = d.copy()
        
        for i in range(len(corr_matrix)):
          for j in range(len(corr_matrix)):
            corr_dist_matrix[i][j]= np.sqrt(np.sum((d[:,i]-d[:,j])**2))
            
        corr_dist_matrix = corr_dist_matrix/np.max(corr_dist_matrix)
        return pd.DataFrame(corr_dist_matrix, index=returns_df.columns, columns=returns_df.columns)

def ESG_metric(ESG_normalized):
    # Compute ESG distance matrix
    esg_dist_matrix = np.abs(ESG_normalized[:, None] - ESG_normalized[None, :])
    return  pd.DataFrame(esg_dist_matrix, index=stocks, columns=stocks)


def compute_clusters(epsilon, num_clusters, print_ = True, annual_returns_dict = annual_returns_dict, ESG_normalized = ESG_normalized):

    # Compute correlation and ESG distance matrix
    global corr_dist_matrix; corr_dist_matrix = corr_metric(returns_df, p)
    global esg_dist_matrix; esg_dist_matrix = ESG_metric(ESG_normalized)

    # Compute the combined correlation and ESG distance matrix
    global combined_dist_matrix; combined_dist_matrix = (1 - epsilon) * corr_dist_matrix + epsilon * esg_dist_matrix    

    # Perform hierarchical clustering and create linkage matrix
    global linkage_matrix; linkage_matrix = linkage(squareform(combined_dist_matrix), method=linkage_method, optimal_ordering=True)

    # Create num_clusters amount of clusters:
    global clusters; clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

    # Create a DataFrame for cluster assignments
    cluster_df = pd.DataFrame({'Stock': combined_dist_matrix.index, 'Cluster': clusters})
    cluster_df = cluster_df.sort_values(by="Cluster").reset_index(drop=True)

    # Compute average ESG score per cluster and variance of ESG between clusters
    esg_df = pd.DataFrame({'Stock': stocks, 'ESG': ESG, 'Cluster': clusters})
    global avg_esg_per_cluster; avg_esg_per_cluster = esg_df.groupby("Cluster")["ESG"].mean()
    global esg_cluster_variance; esg_cluster_variance = avg_esg_per_cluster.var()

    # Create a dataframe for return of clusters per year instead of stocks themselves
    returns_cluster = returns_df.stack().reset_index()
    returns_cluster.columns = ["Year", "Stock", "Return"]
    returns_cluster = returns_cluster.merge(cluster_df, on="Stock")
    avg_returns_per_cluster_per_year = returns_cluster.groupby(["Year", "Cluster"])["Return"].mean().reset_index()

    # Calculate mean return of a cluster and convert to wide format (Years as index, Clusters as columns)
    avg_returns_per_cluster_per_year = returns_cluster.groupby(["Year", "Cluster"])["Return"].mean().reset_index()
    avg_cluster_returns_wide = avg_returns_per_cluster_per_year.pivot(index="Year", columns="Cluster", values="Return")

    # Compute the covariance matrix and total average return of cluster
    avg_cluster_returns_wide = avg_returns_per_cluster_per_year.pivot(index="Year", columns="Cluster", values="Return")
    global cluster_cov_matrix; cluster_cov_matrix = avg_cluster_returns_wide.cov()

    # Average return over entire timeframe
    global avg_return_per_cluster; avg_return_per_cluster = returns_cluster.groupby("Cluster")["Return"].mean().reset_index()

    # Compute condition number of cluster covariance matrix
    global condition_number; condition_number = np.linalg.cond(cluster_cov_matrix)
    # Print the resulting clusters (which stocks; average ESG; average return)
    
    if (print_ == True):
        for cluster_num in range(1, num_clusters + 1):
            cluster_members = cluster_df[cluster_df['Cluster'] == cluster_num]['Stock'].tolist()
            print(
            f"Cluster {cluster_num} (ESG: {avg_esg_per_cluster[cluster_num]:.2f}) "
            f"(Expected return: {avg_return_per_cluster.loc[cluster_num - 1, 'Return']:.2f}): "
            f"{', '.join(cluster_members)}"
            )

        print(f"Condition number of covariance matrix: {condition_number:.2f}")
        print(f"ESG variance across clusters: {esg_cluster_variance:.2f}")

        # Find Optimal Number of Clusters
        methods = ["maxgap", "elbow", "average silhouette"]
        plot_optimal_number_of_clusters(linkage_matrix, combined_dist_matrix, len(stocks), methods, stocks)


def plot_dendrograms(linkage_matrix, epsilon, num_clusters):
    # Plot dendrogram without clusters
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, no_labels=True, leaf_rotation=90)
    plt.title(f"Dendrogram for epsilon = {epsilon} with linkage method: " + linkage_method + 
              " (Note: 0 is only correlation, 1 is only ESG)")
    plt.ylabel("Distance")
    plt.show()


    # Define a function to label the leaves with cluster numbers
    def leaf_label_func(id):
        global label_counter
        label_counter += 1
        return f"Cluster {label_counter}"

    # Plot dendrogram with clusters
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, p=num_clusters, truncate_mode='lastp', 
                                   leaf_rotation=45, leaf_label_func=leaf_label_func)
    plt.title(f"Dendrogram for epsilon = {epsilon} with linkage method: " + linkage_method + 
              " (Note: 0 is only correlation, 1 is only ESG)")
    plt.ylabel("Distance")
    plt.show()

def plot_heatmaps():
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
        sns.heatmap(matrix, ax=ax, cmap=cmap, xticklabels=False, yticklabels=False, annot=False)

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
        
    # Plot original and reordered matrices
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    sns.heatmap(corr_dist_matrix, ax=axes[0, 0], cmap="coolwarm", xticklabels=False, yticklabels=False, annot=False)
    axes[0, 0].set_title("Original Correlation Distance Matrix")

    sns.heatmap(esg_dist_matrix, ax=axes[0, 1], cmap="viridis", xticklabels=False, yticklabels=False, annot=False)
    axes[0, 1].set_title("Original ESG Distance Matrix")

    sns.heatmap(combined_dist_matrix, ax=axes[0, 2], cmap="cividis", xticklabels=False, yticklabels=False, annot=False)
    axes[0, 2].set_title("Original Combined Distance Matrix")

    # Plot reordered heatmaps with cluster borders
    plot_heatmap_with_borders(reordered_corr_dist_matrix, "Reordered Correlation Distance Matrix", axes[1, 0], "coolwarm")
    plot_heatmap_with_borders(reordered_esg_dist_matrix, "Reordered ESG Distance Matrix", axes[1, 1], "viridis")
    plot_heatmap_with_borders(combined_dist_matrix_reordered, "Reordered Combined Distance Matrix", axes[1, 2], "cividis")

    plt.tight_layout()
    plt.show()

def parallel_coordinates_plot(num_clusters_list, epsilon_list):
    results = []
    
    for num_clusters in num_clusters_list:
        for epsilon in epsilon_list:
            compute_clusters(epsilon, num_clusters, False)
            output = optimize_portfolio(avg_return_per_cluster, cluster_cov_matrix, avg_esg_per_cluster)
            # Only include portfolios that satisfy the ESG constraint
            tolerance = 1e-3
            if (output[3] <= ESG_constraint + tolerance):
                results.append({
                    "Number of clusters": num_clusters,
                    "epsilon": epsilon,
                    "ESG variance": esg_cluster_variance,
                    "Expected return": output[1],
                    "Condition number": condition_number,
                    "Risk": output[2]
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df["Condition number (log10)"] = np.log10(results_df["Condition number"])  # Convert to log10 scale

    # For general cases, where axis ranges are dynamically determined
    # fig = go.Figure(data=
    #     go.Parcoords(
    #         line = dict(color = results_df["epsilon"],
    #                    colorscale = "turbo",
    #                    showscale = True,
    #                    cmin = 0,
    #                    cmax = 1,
    #                    colorbar=dict(title="epsilon")),
    #         dimensions = list([
    #             dict(range = [results_df["Number of clusters"].min(), 
    #                           results_df["Number of clusters"].max()],
    #                   label = "Number of clusters", values = results_df["Number of clusters"]),
    #             dict(range = [results_df["Expected return"].min(), 
    #                           results_df["Expected return"].max()],
    #                   label = "Expected return", values = results_df["Expected return"]),
    #             dict(range = [results_df["Risk"].min(), 
    #                           results_df["Risk"].max()],
    #                   label = "Risk", values = results_df["Risk"]),
    #             dict(range = [results_df["Condition number (log10)"].min(), 
    #                           results_df["Condition number (log10)"].max()],
    #                   label = "Condition number (scale log10)", 
    #                   values = results_df["Condition number (log10)"]),
    #             dict(range = [results_df["ESG variance"].min(), 
    #                           results_df["ESG variance"].max()],
    #                   label = "ESG variance", values = results_df["ESG variance"]),
    #             dict(range = [results_df["epsilon"].min(), 
    #                           results_df["epsilon"].max()],
    #                   label = "epsilon", values = results_df["epsilon"])])
    #     )
    # )
    # fig.update_layout(
    #     title=("Portfolios with ESG constraint " + str(ESG_constraint)),
    #     title_font=dict(size=30),
    #     title_x=0.5,
    #     font=dict(size=20)
    # )
    # # Save as an interactive HTML file
    # fig.write_html(f"Portfolios with ESG Constraint {ESG_constraint}.html")
    # fig.show(renderer="browser")
    
    # For our specific results and tailored predefined axis ranges
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = results_df["epsilon"],
                       colorscale = "turbo",
                       showscale = True,
                       cmin = 0,
                       cmax = 1,
                       colorbar=dict(title="epsilon")),
            dimensions = list([
                dict(range = [2, 15],
                     label = "Number of clusters", values = results_df["Number of clusters"]),
                dict(range = [0.16, 0.27],
                    label = "Expected return", values = results_df["Expected return"]),
                dict(range = [0.012, 0.044],
                    label = "Risk", values = results_df["Risk"]),
                dict(range = [0, 1],
                    label = "epsilon", values = results_df["epsilon"])])
        )
    )
    fig.update_layout(
        title=("Risk and return of portfolios with ESG constraint " + str(ESG_constraint)),
        title_font=dict(size=30),
        title_x=0.5,
        font=dict(size=20)
    )
    # Save as an interactive HTML file
    fig.write_html(f"Risk and return of portfolios with ESG constraint {ESG_constraint}.html")
    fig.show(renderer="browser")
    
    
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = results_df["epsilon"],
                       colorscale = "turbo",
                       showscale = True,
                       cmin = 0,
                       cmax = 1,
                       colorbar=dict(title="epsilon")),
            dimensions = list([
                dict(range = [2, 15],
                     label = "Number of clusters", values = results_df["Number of clusters"]),
                dict(range = [0 , 8],
                     label = "Condition number (scale log10)", 
                     values = results_df["Condition number (log10)"]),
                dict(range = [0, 125],
                     label = "ESG variance", values = results_df["ESG variance"]),
                dict(range = [0, 1],
                    label = "epsilon", values = results_df["epsilon"])])
        )
    )
    fig.update_layout(
        title = ("Condition number and ESG variance of portfolios with ESG constraint " + str(ESG_constraint)),
        title_font=dict(size=30),
        title_x=0.5,
        font=dict(size=20)
    )
    # Save as an interactive HTML file
    fig.write_html(f"Condition number and ESG variance of portfolios with ESG constraint {ESG_constraint}.html")
    fig.show(renderer="browser")

def optimize_portfolio(avg_return_per_cluster, cluster_cov_matrix, avg_esg_per_cluster):
    # Constraints for optimization problem
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested constraint
        {'type': 'ineq', 'fun': lambda w: ESG_constraint - w @ avg_esg_per_cluster.values}  # ESG should be below λ
    ]
    # Bounds: No shorting (long-only portfolio)
    bounds = [(0, 1) for _ in range(len(avg_return_per_cluster))]
    
    init_weights = np.ones(len(avg_return_per_cluster)) / len(avg_return_per_cluster)
    
    result = minimize(objective, init_weights, args=(avg_return_per_cluster['Return'].values, cluster_cov_matrix, risk_aversion),
                     constraints=constraints, bounds=bounds, method='SLSQP')
    optimal_weights = result.x
    
    # Compute the portfolio expected return (weighted average)
    portfolio_expected_return = np.dot(optimal_weights, avg_return_per_cluster["Return"].values)
    
    # Compute the portfolio risk
    portfolio_risk = optimal_weights @ cluster_cov_matrix @ optimal_weights
    
    # Compute the portfolio ESG score (weighted average)
    portfolio_esg_score = np.dot(optimal_weights, avg_esg_per_cluster.values)
    
    return optimal_weights, portfolio_expected_return, portfolio_risk, portfolio_esg_score
    
    
# Declare for the optimzation problem
def objective(weights, returns, cov_matrix, risk_aversion):
    return - (weights @ returns - (risk_aversion / 2) * (weights @ cov_matrix @ weights))

# Parameters and constraints for optimization
global risk_aversion; risk_aversion = 5
global ESG_constraint; ESG_constraint = 20  # Adjust ESG constraint (weighted ESG score must be lower than the constraint)

global p; p = 0 # Choose metric
global linkage_method; linkage_method = "ward"  # Choose "single", "complete", "average", "ward", etc.

# Single numerical experiment
epsilon = 0.15
num_clusters = 8 # Choose number of clusters
compute_clusters(epsilon, num_clusters)
plot_dendrograms(linkage_matrix, epsilon, num_clusters)
plot_heatmaps()
output = optimize_portfolio(avg_return_per_cluster, cluster_cov_matrix, avg_esg_per_cluster)

# Print portfolio results considering weights, ESG, expected return and risk
np.set_printoptions(suppress=True, precision=2)
print("Optimal weights per cluster:\n", output[0])
print(f"Weighted expected return of portfolio: {output[1]:.2f}")
print(f"Risk of portfolio: {output[2]:.2f}")
print(f"Weighted ESG score of portfolio: {output[3]:.2f}")

# Perform numerical experiments and creating a parallel coordinates plot
num_clusters_list = list(range(2, 16, 1))
epsilon_list = np.arange(0, 1.05, 0.05)
parallel_coordinates_plot(num_clusters_list, epsilon_list)
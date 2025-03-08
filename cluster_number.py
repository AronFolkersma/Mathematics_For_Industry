import numpy as np
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

def build_tree(linkage_matrix, labels=None):
    n = linkage_matrix.shape[0] + 1  # Number of original observations
    
    def _build_tree(node_id):
        if node_id < n:
            return {
                "id": node_id,
                "name": labels[node_id] if labels else str(node_id)
            }  # Leaf node
        
        left_child = int(linkage_matrix[node_id - n, 0])
        right_child = int(linkage_matrix[node_id - n, 1])
        distance = linkage_matrix[node_id - n, 2]
        size = int(linkage_matrix[node_id - n, 3])
        n_clusters = 2*n - (node_id + 1)
        
        return {
            "id": node_id,
            "clusters": n_clusters,
            "distance": distance,
            "size": size,
            "children": (_build_tree(left_child), _build_tree(right_child))
        }
    
    return _build_tree(2 * n - 2)  # Root node is at index 2n-2

def collapse(tree):
    if tree.get("name"): return [tree["name"]]            # leaf
    
    left_child, right_child = tree["children"]            # node
    return collapse(left_child) + collapse(right_child)

def filter_empties(array):
    return [x for x in array if x != []]

def collapse_to_i_clusters(tree, i):
    if tree.get("name"): return [[tree["name"]]]                              # leaf
    
    if tree["clusters"] >= i:                                               # node 
        return [collapse(tree)]
    
    left, right = tree["children"]

    return collapse_to_i_clusters(left, i) + collapse_to_i_clusters(right, i)

def compute_wss(cluster_indices, distance_matrix):
    if len(cluster_indices) < 2:
        return 0  # WSS is zero if the cluster has only one or no elements
    
    distance_matrix = np.array(distance_matrix)

    # Compute the centroid (average of points in the cluster)
    cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
    centroid_distances = cluster_distances.mean(axis=0)  # Mean distance to all points
    
    # Compute WSS as the sum of squared distances to the centroid
    wss = np.sum((cluster_distances - centroid_distances) ** 2)
    
    return wss

def clusters_to_labels(inputs, clusters):
    # Create a dictionary mapping each input value to its cluster index
    label_map = {}
    for cluster_idx, cluster in enumerate(clusters):
        for value in cluster:
            label_map[value] = cluster_idx
    
    # Generate the output array with labels for each input
    labels = [label_map[value] for value in inputs]
    
    return labels

def cluster_to_index(cluster, tickers):
    indices = []
    for ticker in cluster:
        indices.append(tickers.index(ticker))
    return indices


def dist_from_number_of_clusters(tree, i):
    if tree.get("name"): return 0
    if tree["clusters"] == i: return tree["distance"]

    left, right = tree["children"]

    return dist_from_number_of_clusters(left, i) + dist_from_number_of_clusters(right, i)

def biggest_drop_index(arr):
    if len(arr) < 2:
        return -1  # No drop possible if there are fewer than 2 elements
    
    max_drop = 0
    max_index = -1
    
    for i in range(len(arr) - 1):
        drop = arr[i] - arr[i + 1]
        if drop > max_drop:
            max_drop = drop
            max_index = i+1  # Store the index of the first element in the biggest drop
    
    return max_index

def optimal_number_of_clusters(linkage_matrix, distance_matrix, n_stocks:int, method:str, tickers):
    tree = build_tree(linkage_matrix, tickers)
    
    ns = np.arange(1, n_stocks+1)
    vs = np.zeros(len(ns))

    for n in ns:
        clusters = collapse_to_i_clusters(tree, n)
        if method == "maxgap":
            vs[n-1] = dist_from_number_of_clusters(tree, n)
        elif method == "elbow":
            vs[n-1] = sum(compute_wss(cluster_to_index(cluster, tickers), distance_matrix) for cluster in clusters)
        elif method == "average silhouette":
            labels = clusters_to_labels(tickers, clusters)
            if len(set(labels)) > 1 and len(set(labels)) < n_stocks:  # Ensure there is more than one cluster
                vs[n-1] = silhouette_score(distance_matrix, labels=labels, metric="precomputed")
            else:
                vs[n-1] = 0  
        elif method == "gap statistic":
            vs[n-1] = 0
                
    if method == "maxgap":
        optimal_n = ns[biggest_drop_index(vs)]
    elif method == "elbow":
        optimal_n = KneeLocator(ns, vs, curve="convex", direction="decreasing").knee
    elif method == "average silhouette":
        optimal_n = ns[np.argmax(vs)]

    return ns, vs, optimal_n

import matplotlib.pyplot as plt

def plot_optimal_number_of_clusters(linkage_matrix, distance_matrix, n_stocks: int, methods, tickers):
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5))
    
    if len(methods) == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        ns, vs, optimal_n = optimal_number_of_clusters(linkage_matrix, distance_matrix, n_stocks, method, tickers)

        ax.axvline(x=optimal_n , ls='--', color='r', label=f"n={optimal_n}"
)

        if method == "elbow":
            kn = KneeLocator(ns, vs, curve="convex", direction="decreasing")
            ax.axvline(x=kn.knee, ls='--', color='r', label=f"Elbow Point: n={kn.knee}")
        
        ax.plot(ns, vs, marker='o', label=method)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Score")
        ax.set_title(f"Method: {method}")
        ax.legend()
        ax.grid()
    
    plt.tight_layout()
    plt.show()



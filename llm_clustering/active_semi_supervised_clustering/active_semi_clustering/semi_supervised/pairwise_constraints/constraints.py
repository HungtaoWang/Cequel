from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from llm_clustering.active_semi_supervised_clustering.active_semi_clustering.exceptions import InconsistentConstraintsException

# Taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
def preprocess_constraints(ml, cl, n):
    "Create a graph of constraints for both must- and cannot-links"

    print("Preprocessing constraints")
    # Represent the graphs using adjacency-lists
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    # Run DFS from each node to get all the graph's components
    # and add an edge for each pair of nodes in the component (create a complete graph)
    # See http://www.techiedelight.com/transitive-closure-graph/ for more details
    visited = [False] * n
    neighborhoods = []
    for i in tqdm(range(n)):
        if not visited[i] and ml_graph[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2 and (x1 not in cl_graph[x2] and x2 not in cl_graph[x1]):
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    for (i, j) in tqdm(cl):
        for x in ml_graph[i]:
            if x not in ml_graph[j] and j not in ml_graph[x]:
                add_both(cl_graph, x, j)

        for y in ml_graph[j]:
            if y not in ml_graph[i] and i not in ml_graph[y]:
                add_both(cl_graph, i, y)

        for x in ml_graph[i]:
            for y in ml_graph[j]:
                if x not in ml_graph[y] and y not in ml_graph[x]:
                    add_both(cl_graph, x, y)

    _ = """
    for (i, j) in tqdm(cl[:150]):
        timer1 = time.perf_counter()
        for x in ml_graph[i]:
            timer1_2 = time.perf_counter()
            if x not in ml_graph[j] and j not in ml_graph[x]:
                timer1_3 = time.perf_counter()
                timer_dict["check_ml"] += timer1_3 - timer1_2
                add_both(cl_graph, x, j)
                timer_dict["add_both_cl"] += time.perf_counter() - timer1_3
        timer2 = time.perf_counter()
        for y in ml_graph[j]:
            timer2_2 = time.perf_counter()
            if y not in ml_graph[i] and i not in ml_graph[y]:
                timer2_3 = time.perf_counter()
                timer_dict["check_ml"] += timer2_3 - timer2_2
                add_both(cl_graph, i, y)
                timer_dict["add_both_cl"] += time.perf_counter() - timer2_3
        timer3 = time.perf_counter()
        for x in ml_graph[i]:
            for y in ml_graph[j]:
                timer3_2 = time.perf_counter()
                if x not in ml_graph[y] and y not in ml_graph[x]:
                    timer3_3 = time.perf_counter()
                    timer_dict["check_ml"] += timer3_3 - timer3_2
                    add_both(cl_graph, x, y)
                    timer_dict["add_both_cl"] += time.perf_counter() - timer3_3
        timer4 = time.perf_counter()
        timer_dict["head_ml_expansion"] += timer2 - timer1
        timer_dict["tail_ml_expansion"] += timer3 - timer2
        timer_dict["both_sides_expansion"] += timer4 - timer3

    timer_dict = {"add_both_cl": 0.0,
                    "check_ml": 0.0,
                    "head_ml_expansion": 0.0,
                    "tail_ml_expansion": 0.0,
                    "both_sides_expansion": 0.0}    

    """

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise InconsistentConstraintsException('Inconsistent constraints between {} and {}'.format(i, j))

    return ml_graph, cl_graph, neighborhoods


def preprocess_constraints_no_transitive_closure(ml, cl, n):
    "Create a graph of constraints for both must- and cannot-links"

    # Represent the graphs using adjacency-lists
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    return ml_graph, cl_graph

def compute_weighted_degree(X):
    """
    Compute the weighted degree for each node based on the provided formula.
    Parameters:
    - X: The normalized embeddings matrix (each row is a normalized vector).
    Returns:
    - z: A vector representing the weighted degree of each node.
    """
    # Step 1: Compute the dot product matrix for all vectors
    dot_matrix = np.dot(X, X.T)  # This gives us the pairwise dot products
    
    # Step 2: Replace negative dot products with 0
    dot_matrix[dot_matrix < 0] = 1e-10  # Set all negative dot products to 0
    
    # Step 3: Compute the weighted degree by summing each row
    z = np.sum(dot_matrix, axis=1)  # Sum over each row to get the weighted degree for each node
    
    return z


def compute_weight(x_a, x_b, d_a, d_b, T, weight_method="cosine"):
    """
    Compute weight for a constraint between two points, considering degree information.
    Parameters:
    - x_a, x_b: Feature vectors of the two points.
    - d_a, d_b: Weighted degrees of the two points.
    - T: Array of all degrees d(x), used to compute \sum_{t_x \in T} 1/d(x).
    - weight_method: The method used to calculate the weight. Supported methods:
        - "cosine": Cosine similarity.
        - "sqrt_cosine": sqrt(cosine).
        - "log_cosine": log(cosine + 1).
        - "degree": 1/d(a) + 1/d(b).
        - "sqrt_degree": sqrt(1/d(a) + 1/d(b)).
        - "log_degree_ratio": log((d(a) * d(b) / w(t_a, t_b)) * sum(1 / d(x)) + 1).
    Returns:
    - weight: Calculated weight for the constraint.
    """
    # Compute cosine similarity
    # cosine_similarity = np.dot(x_a, x_b) / (np.linalg.norm(x_a) * np.linalg.norm(x_b) + 1e-10)
    cosine_similarity = np.dot(x_a, x_b)

    if cosine_similarity < 0:
        cosine_similarity = 1e-10

    if weight_method == "cosine":
        return cosine_similarity

    elif weight_method == "sqrt_cosine":
        return np.sqrt(cosine_similarity)

    elif weight_method == "log_cosine":
        return np.log(cosine_similarity + 1)

    elif weight_method == "degree":
        return 1 / d_a + 1 / d_b

    elif weight_method == "sqrt_degree":
        return np.sqrt(1 / d_a + 1 / d_b)

    elif weight_method == "pmi_log":
        # numerator = cosine_similarity * np.sum(d_a)
        numerator = cosine_similarity * np.sum(T)
        denominator = d_a * d_b
        return np.log(numerator / denominator + 1)

    elif weight_method == "log_degree_ratio":
        # Compute the summation term \sum_{t_x \in T} 1 / d(x)
        degree_sum = np.sum(1 / T)  # T is the array of all degrees d(x)
        # ratio = (d_a * d_b) / (cosine_similarity + 1e-10)  # Avoid division by zero
        ratio = (d_a * d_b) / (cosine_similarity)  # Avoid division by zero
        return np.log(ratio * degree_sum + 1)

    else:
        raise ValueError(f"Unsupported weight computation method: {weight_method}")


def preprocess_constraints_with_weights(ml, cl, n, X, weight_method="cosine"):
    """
    Create a graph of constraints for both must- and cannot-links, and calculate weights.
    Parameters:
    - ml: List of must-link constraints [(i, j), ...].
    - cl: List of cannot-link constraints [(i, j), ...].
    - n: Total number of data points.
    - X: Data matrix (n_samples, n_features), used to calculate weights.
    - weight_method: Method to compute weights, default is "cosine".
    Returns:
    - ml_graph: Adjacency list for must-link constraints.
    - cl_graph: Adjacency list for cannot-link constraints.
    - ml_weights: Dictionary of weights for must-link constraints.
    - cl_weights: Dictionary of weights for cannot-link constraints.
    """
    # Compute weighted degrees
    d = compute_weighted_degree(X)
    T = d  # Use the degree array directly for summation in log_degree_ratio

    # Initialize adjacency lists and weight dictionaries
    ml_graph, cl_graph = {}, {}
    ml_weights, cl_weights = {}, {}

    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    # Helper function to add bidirectional edges
    def add_both(d_graph, w_graph, i, j, weight):
        d_graph[i].add(j)
        d_graph[j].add(i)
        w_graph[(i, j)] = weight
        w_graph[(j, i)] = weight

    # Compute weights for each must-link constraint
    for (i, j) in ml:
        weight = compute_weight(X[i], X[j], d[i], d[j], T, weight_method)
        add_both(ml_graph, ml_weights, i, j, weight)

    # Compute weights for each cannot-link constraint
    for (i, j) in cl:
        weight = compute_weight(X[i], X[j], d[i], d[j], T, weight_method)
        add_both(cl_graph, cl_weights, i, j, weight)

    def normalize_weights_ml(weights):
        values = np.array(list(weights.values())).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0.01, 0.1))
        normalized_values = scaler.fit_transform(values).flatten()
        return {
            key: normalized_values[i]
            for i, key in enumerate(weights.keys())
        }
    def normalize_weights_cl(weights):
        values = np.array(list(weights.values())).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 0.01))
        normalized_values = scaler.fit_transform(values).flatten()
        return {
            key: normalized_values[i]
            for i, key in enumerate(weights.keys())
        }

    normalized_ml_weights = normalize_weights_ml(ml_weights)
    normalized_cl_weights = normalize_weights_cl(cl_weights)

    return ml_graph, cl_graph, normalized_ml_weights, normalized_cl_weights

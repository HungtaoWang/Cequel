import math

from collections import Counter

from sklearn.preprocessing import MinMaxScaler

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from scipy import spatial

import scipy as sp
from sklearn.cluster import KMeans

import time

import argparse



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--corpus_name', type=str, 
                        choices=['BBC_News', 'Tweet', 'Bank77', 'Reddit', 'CLINC', 'Massive'],
                        default='BBC_News')
    parser.add_argument('--selection', type=str, choices=['edge', 'triangle'], default='triangle')

    parser.add_argument('--encoder_name', type=str, choices=['instructor', 'sentencebert'], default='instructor')
    parser.add_argument('--clustering', type=str, 
                        choices=['WeightedPCKMeans', 'WeightedConstrainedSpectralClustering'],
                        default='WeightedPCKMeans')
    parser.add_argument('--mode', type=str, choices=['max-sum'], default='max-sum')
    parser.add_argument('--weight_method', type=str, 
                        choices=['cosine', 'sqrt_cosine', 'log_cosine', 'degree', 'sqrt_degree', 'pmi_log', 'log_degree_ratio'],
                        default='log_degree_ratio')
    parser.add_argument('--eigen', type=float, default=0.1)

    return parser.parse_args()


def compute_weighted_degree(X):
    """
    Compute the weighted degree for each node based on the provided formula.
    Parameters:
    - X: The normalized embeddings matrix (each row is a normalized vector).
    Returns:
    - z: A vector representing the weighted degree of each node.
    """

    dot_matrix = np.dot(X, X.T)  

    dot_matrix[dot_matrix < 0] = 1e-10  # Set all negative dot products to 0
    
    z = np.sum(dot_matrix, axis=1)  
    
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
        numerator = cosine_similarity * np.sum(T)
        denominator = d_a * d_b
        return np.log(numerator / denominator + 1)

    elif weight_method == "log_degree_ratio":
        degree_sum = np.sum(1 / T)  # T is the array of all degrees d(x)
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
        scaler = MinMaxScaler(feature_range=(0.5, 1.5))
        normalized_values = scaler.fit_transform(values).flatten()
        return {
            key: normalized_values[i]
            for i, key in enumerate(weights.keys())
        }
    def normalize_weights_cl(weights):
        values = np.array(list(weights.values())).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0.5, 1.5))
        normalized_values = scaler.fit_transform(values).flatten()
        return {
            key: normalized_values[i]
            for i, key in enumerate(weights.keys())
        }

    normalized_ml_weights = normalize_weights_ml(ml_weights)
    normalized_cl_weights = normalize_weights_cl(cl_weights)

    return ml_graph, cl_graph, normalized_ml_weights, normalized_cl_weights


def create_affinity_matrix(X):
    tree = spatial.KDTree(X)
    dist, idx = tree.query(X, k=16)
    
    idx = idx[:,1:]
    
    nb_data, _ = X.shape
    A = np.zeros((nb_data, nb_data))
    for i, j in zip(np.arange(nb_data), idx):
        A[i, j] = 1
    A = np.maximum(A.T, A)

    return A

def create_constraint_matrix_weight(ml, cl, num_points, X, weight_method):

    Q = np.zeros((num_points, num_points), dtype=int)
    Q[np.arange(Q.shape[0]), np.arange(Q.shape[0])] = 1
    
    ml_graph, cl_graph, ml_weights, cl_weights = preprocess_constraints_with_weights(
            ml, cl, X.shape[0], X, weight_method=weight_method
        )
    
    for (i, j), weight in ml_weights.items():
        Q[i, j] = weight
        Q[j, i] = weight  

    for (i, j), weight in cl_weights.items():
        Q[i, j] = -weight
        Q[j, i] = -weight  
    
    return Q


def CSCK(X, n_clusters, normalize_vectors, constraints, z, weight_method, clustering, eigen):
    K = n_clusters

    ml, cl = constraints

    if normalize_vectors:
        sc = preprocessing.StandardScaler()
        sc.fit(X)
        X_norm = sc.transform(X)
    else:
        X_norm = X

    A = create_affinity_matrix(X_norm)
        
    if clustering == "WeightedConstrainedSpectralClustering":
        Q = create_constraint_matrix_weight(ml, cl, X.shape[0], X, weight_method)

    D = np.diag(np.sum(A, axis=1))
    vol = np.sum(A)

    D_norm = np.linalg.inv(np.sqrt(D))
    L_norm = np.eye(*A.shape) - D_norm.dot(A.dot(D_norm))
    Q_norm = D_norm.dot(Q.dot(D_norm))

    alpha = eigen * sp.linalg.svdvals(Q_norm)[K]
    Q1 = Q_norm - alpha * np.eye(*Q_norm.shape)
    
    val, vec = sp.linalg.eig(L_norm, Q1)

    vec = vec[:,val >= 0]
    vec_norm = (vec / np.linalg.norm(vec, axis=0)) * np.sqrt(vol)

    costs = np.multiply(vec_norm.T.dot(L_norm), vec_norm.T).sum(axis=1)
    ids = np.where(costs > 1e-10)[0]
    min_idx = np.argsort(costs[ids])[0:K]
    min_v = vec_norm[:,ids[min_idx]]

    u = D_norm.dot(min_v)
   
    model = KMeans(n_clusters=K).fit(u)
    labels = model.labels_
    
    return labels

datasets = [
    {"Corpus": "BBC_News", "Texts": 2225, "Clusters": 5, "Task": "News"},
    {"Corpus": "Tweet", "Texts": 2472, "Clusters": 89, "Task": "Tweet"},
    {"Corpus": "Bank77", "Texts": 3080, "Clusters": 77, "Task": "Intent"},
    {"Corpus": "Reddit", "Texts": 3217, "Clusters": 50, "Task": "Topic"},
    {"Corpus": "CLINC", "Texts": 4500, "Clusters": 10, "Task": "Domain"},
    {"Corpus": "Massive", "Texts": 11514, "Clusters": 18, "Task": "Type"}
]

args = parse_args()
corpus_name = args.corpus_name
selection = args.selection
encoder_name = args.encoder_name
clustering = args.clustering
mode = args.mode
weight_method = args.weight_method
eigen = args.eigen


def get_dataset_info(corpus_name):
    for dataset in datasets:
        if dataset["Corpus"] == corpus_name:
            return dataset
    return None

target_dataset = get_dataset_info(corpus_name)

corpus_name = target_dataset["Corpus"]
task_name = target_dataset["Task"]
doc_number = target_dataset["Texts"]
cluster_number = target_dataset["Clusters"]


edge_prompt = f"Cluster {corpus_name} docs by whether they are the same {task_name.lower()} type. For each pair, respond with Yes or No without explanation."
triangle_prompt = f"Cluster {corpus_name} docs by whether they are the same {task_name.lower()} type. For each triangle, respond with a, b, c, d, or e without explanation."

if selection == "edge":
    prompt = edge_prompt
elif selection == "triangle":
    prompt = triangle_prompt


from llm_clustering.wrappers import LLMPairwiseClustering

from llm_clustering.dataloaders import load_corpus


if encoder_name == "instructor":
    if corpus_name == "BBC_News":
        cache_path = f"embeddings/bbc_instructor-large.pt"
        file_path = f"data/bbc.csv"
    elif corpus_name == "Tweet":
        cache_path = f"embeddings/tweet_instructor-large.pt"
        file_path = f"data/tweet.csv"
    elif corpus_name == "Bank77":
        cache_path = f"embeddings/bank77_instructor-large.pt"
        file_path = f"data/bank77.jsonl"
    elif corpus_name == "Reddit":
        cache_path = f"embeddings/reddit_instructor-large.pt"
        file_path = f"data/reddit.jsonl"
    elif corpus_name == "CLINC":
        cache_path = f"embeddings/clinic_instructor-large.pt"
        file_path = f"data/clinic.jsonl"
    elif corpus_name == "Massive":
        cache_path = f"embeddings/massive_scenario_instructor-large.pt"
        file_path = f"data/massive_scenario.jsonl"

if encoder_name == "sentencebert":
    if corpus_name == "BBC_News":
        cache_path = f"embeddings/bbc_sentence_bert.pt"
        file_path = f"data/bbc.csv"
    elif corpus_name == "Tweet":
        cache_path = f"embeddings/tweet_sentence_bert.pt"
        file_path = f"data/tweet.csv"
    elif corpus_name == "Bank77":
        cache_path = f"embeddings/bank77_sentence_bert.pt"
        file_path = f"data/bank77.jsonl"
    elif corpus_name == "Reddit":
        cache_path = f"embeddings/reddit_sentence_bert.pt"
        file_path = f"data/reddit.jsonl"
    elif corpus_name == "CLINC":
        cache_path = f"embeddings/clinic_sentence_bert.pt"
        file_path = f"data/clinic.jsonl"
    elif corpus_name == "Massive":
        cache_path = f"embeddings/massive_scenario_sentence_bert.pt"
        file_path = f"data/massive_scenario.jsonl"



features, labels, documents = load_corpus(cache_path, file_path, encoder_name)

prompt_suffix = f"correspond to the same general {task_name.lower()} type?"
text_type = f"{task_name}"



if selection == "edge":
    max_feedback = doc_number // 2
elif selection == "triangle":
    max_feedback = doc_number // 3

data = corpus_name

print('data:')
print(data)
print('mode:')
print(mode)

cache_file = f"cache/{encoder_name}/{selection}/query/{data}-{mode}.json"

cluster_assignments, constraints = LLMPairwiseClustering(features, documents, cluster_number, prompt, text_type, prompt_suffix, max_feedback_given=max_feedback, pckmeans_w=0.01, cache_file=cache_file, constraint_selection_algorithm="SimilarityFinder", kmeans_init="k-means++", mode=mode, data=data, selection=selection, clustering=clustering, weight_method=weight_method, encoder_name=encoder_name)

if clustering == "WeightedConstrainedSpectralClustering":
    print("CSCK")
    start_time = time.time()

    assignments = CSCK(features, cluster_number, False, constraints, np.array(labels), weight_method, clustering, eigen)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Code execution time: {execution_time:.4f} seconds")



import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

# ===== Helper: ACC (Clustering Accuracy) =====
def cluster_acc(true_labels, predicted_labels):
    """
    Computes Clustering Accuracy (ACC) using the Hungarian algorithm for optimal assignment.
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Create cost matrix for Hungarian algorithm
    max_label = max(predicted_labels.max(), true_labels.max()) + 1
    matrix = np.zeros((max_label, max_label), dtype=int)
    for t, p in zip(true_labels, predicted_labels):
        matrix[t, p] += 1

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(-matrix)
    acc = matrix[row_ind, col_ind].sum() / len(true_labels)
    return acc

if clustering == "WeightedPCKMeans":
    # print(f"Accuracy: {cluster_acc(np.array(cluster_assignments), np.array(labels))}")
    labels = np.array(labels)
    cluster_assignments = np.array(cluster_assignments)

    acc = cluster_acc(labels, cluster_assignments)
    nmi = normalized_mutual_info_score(labels, cluster_assignments)

    print(f"Args parsed:\n"
          f"corpus_name = {corpus_name}\n"
          f"selection = {selection}\n"
          f"encoder_name = {encoder_name}\n"
          f"clustering = {clustering}\n"
          f"mode = {mode}\n"
          f"weight_method = {weight_method}\n"
          f"eigen = {eigen}\n")
    print(f"Accuracy: {acc:.4f}")
    print(f"NMI: {nmi:.4f}")


if clustering == "WeightedConstrainedSpectralClustering":
    print("spectral")
    labels = np.array(labels)
    assignments = np.array(assignments)

    acc = cluster_acc(labels, assignments)
    nmi = normalized_mutual_info_score(labels, assignments)

    print(f"Args parsed:\n"
          f"corpus_name = {corpus_name}\n"
          f"selection = {selection}\n"
          f"encoder_name = {encoder_name}\n"
          f"clustering = {clustering}\n"
          f"mode = {mode}\n"
          f"weight_method = {weight_method}\n"
          f"eigen = {eigen}\n")
    print(f"Accuracy: {acc:.4f}")
    print(f"NMI: {nmi:.4f}")

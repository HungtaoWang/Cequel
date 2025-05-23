import os
import time
import json
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

# ===== Helper: ACC (Clustering Accuracy) =====
def clustering_accuracy(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    max_label = max(predicted_labels.max(), true_labels.max()) + 1
    matrix = np.zeros((max_label, max_label), dtype=int)
    for t, p in zip(true_labels, predicted_labels):
        matrix[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-matrix)
    acc = matrix[row_ind, col_ind].sum() / len(true_labels)
    return acc

# ===== Step 1: Read Data =====
def read_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith('.jsonl'):
        texts, labels, label_mapping = [], [], {}
        current_index = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                texts.append(data['input'])
                if data['label'] not in label_mapping:
                    label_mapping[data['label']] = current_index
                    current_index += 1
                labels.append(label_mapping[data['label']])
        return texts, labels, label_mapping

    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        if 'text' not in df.columns:
            raise ValueError(f"[ERROR] CSV file {file_path} must contain a 'text' column.")

        label_column = None
        for col in ['label', 'cluster']:
            if col in df.columns:
                label_column = col
                break

        if not label_column:
            raise ValueError(f"[ERROR] CSV file {file_path} must contain a valid label column (e.g., 'label', 'cluster').")

        texts = df['text'].tolist()
        labels = df[label_column].astype(int).tolist()
        return texts, labels, None

# ===== Step 2: Load Embeddings =====
def load_embeddings(embedding_folder):
    all_embeddings = {}
    for filename in os.listdir(embedding_folder):
        if filename.endswith(".pt"):
            filepath = os.path.join(embedding_folder, filename)

            if "go_emo" in filename:
                dataset_name = "go_emo"
                model_name = filename.replace("go_emo_", "").replace(".pt", "")
            elif "few_nerd" in filename:
                dataset_name = "few_nerd"
                model_name = filename.replace("few_nerd_", "").replace(".pt", "")
            elif "massive_scenario" in filename:
                dataset_name = "massive_scenario"
                model_name = filename.replace("massive_scenario_", "").replace(".pt", "")
            elif "mtop_domain" in filename:
                dataset_name = "mtop_domain"
                model_name = filename.replace("mtop_domain_", "").replace(".pt", "")
            elif "few_event" in filename:
                dataset_name = "few_event"
                model_name = filename.replace("few_event", "").replace(".pt", "")
            else:
                split_name = filename.split("_", 1)
                dataset_name = split_name[0]
                model_name = split_name[1].replace(".pt", "")

            if dataset_name not in all_embeddings:
                all_embeddings[dataset_name] = {}

            print(f"[INFO] Loading embedding: {filename}")
            embedding = torch.load(filepath, map_location=torch.device('cuda:1'))
            all_embeddings[dataset_name][model_name] = embedding
    return all_embeddings

# ===== Step 3: Clustering =====
def perform_clustering(embeddings, true_labels, n_clusters, method="kmeans", seeds=10, return_all=False):
    acc_list = []
    nmi_list = []
    times = []
    details = []

    random_seeds = np.random.choice(10000, seeds, replace=False)

    for i, seed in enumerate(random_seeds):
        print(f"[INFO] Running {method} clustering | Seed {i+1}/{seeds} | Random State: {seed}")
        start_time = time.time()
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=seed)
        elif method == "spectral":
            model = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=seed)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        predicted_labels = model.fit_predict(embeddings)
        clustering_time = time.time() - start_time

        acc = clustering_accuracy(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        acc_list.append(acc)
        nmi_list.append(nmi)
        times.append(clustering_time)

        details.append({
            "Seed": int(seed),
            "ACC": acc,
            "NMI": nmi,
            "Time": clustering_time
        })

    result_summary = {
        "Mean_ACC": np.mean(acc_list),
        "Std_ACC": np.std(acc_list),
        "Mean_NMI": np.mean(nmi_list),
        "Std_NMI": np.std(nmi_list),
        "Mean_Time": np.mean(times),
        "Std_Time": np.std(times)
    }

    if return_all:
        return result_summary, details
    else:
        return result_summary

# ===== Main Execution =====
if __name__ == "__main__":
    embedding_folder = "./embeddings"
    data_folder = "./data"

    dataset_clusters = {
        "bbc": 5,
        "tweet": 89,
        "banking77": 77,
        "reddit": 50,
        "clinc": 10,
        "massive_scenario": 18
    }

    all_embeddings = load_embeddings(embedding_folder)
    print("[INFO] Embedding loading complete.")
    final_results = []
    seed_level_results = []

    for dataset_name, models in all_embeddings.items():
        data_file_jsonl = os.path.join(data_folder, f"{dataset_name}.jsonl")
        data_file_csv = os.path.join(data_folder, f"{dataset_name}.csv")

        if os.path.exists(data_file_jsonl):
            data_file = data_file_jsonl
        elif os.path.exists(data_file_csv):
            data_file = data_file_csv
        else:
            print(f"[ERROR] Data file not found for dataset: {dataset_name}")
            continue

        texts, true_labels, _ = read_data(data_file)
        n_clusters = dataset_clusters[dataset_name]

        for model_name, embedding in models.items():
            print(f"[INFO] Processing Dataset: {dataset_name} | Model: {model_name}")
            embeddings = embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else np.array(embedding)
            normalized_embeddings = normalize(embeddings, norm='l2')

            kmeans_summary, kmeans_details = perform_clustering(normalized_embeddings, true_labels, n_clusters, method="kmeans", seeds=20, return_all=True)
            spectral_summary, spectral_details = perform_clustering(normalized_embeddings, true_labels, n_clusters, method="spectral", seeds=20, return_all=True)

            final_results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Method": "KMeans++",
                **kmeans_summary
            })
            final_results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Method": "Spectral",
                **spectral_summary
            })

            for record in kmeans_details:
                seed_level_results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Method": "KMeans++",
                    **record
                })
            for record in spectral_details:
                seed_level_results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Method": "Spectral",
                    **record
                })

    results_df = pd.DataFrame(final_results)
    results_df.to_csv("clustering_results_summary.csv", index=False)

    seed_df = pd.DataFrame(seed_level_results)
    seed_df.to_csv("clustering_results_seeds.csv", index=False)

    print("[INFO] Summary results saved to 'clustering_results_summary.csv'")
    print("[INFO] Seed-level results saved to 'clustering_results_seeds.csv'")

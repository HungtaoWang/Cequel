from collections import defaultdict, namedtuple

from datasets import load_dataset as load_dataset_hf
import os
import json
import numpy as np
import pandas as pd
import pickle
import random
import torch
from transformers import AutoModel, AutoTokenizer

from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from sklearn import datasets, metrics

from few_shot_clustering.cmvc.CMVC_main_opiec import CMVC_Main as CMVC_Main_opiec
from few_shot_clustering.cmvc.CMVC_main_reverb45k import CMVC_Main as CMVC_Main_reverb
from few_shot_clustering.cmvc.helper import invertDic

from few_shot_clustering.cmvc import preprocessing

from sklearn.preprocessing import normalize

def normalize_rows(X):
    """
    Normalize each row of matrix X to have unit L2 norm.

    Parameters:
    - X: 2D numpy array where each row is a vector.

    Returns:
    - Normalized matrix X with each row having L2 norm equal to 1.
    """
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)  # Compute L2 norm of each row
    normalized_X = X / row_norms  # Divide each row by its norm
    return normalized_X

def check_tensor_device(var):
    if torch.is_tensor(var):
        print(f"The variable is a Tensor on device: {var.device}")
    else:
        print("The variable is not a Tensor")

def read_data(file_path):
    """
    Reads a single file (either .jsonl or .csv) and extracts text and label.
    If the file is .jsonl, it creates a label-to-index mapping.
    CSV files automatically skip the header row.
    
    Parameters:
        file_path (str): Path to the file to process.
    
    Returns:
        texts (list): List of text entries.
        labels (list): List of labels as indices for jsonl, or raw for csv.
        label_mapping (dict): Mapping from label strings to indices (for jsonl).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Process .jsonl file
    if file_path.endswith('.jsonl'):
        texts = []
        labels = []
        label_mapping = {}
        current_index = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                texts.append(data['input'])
                
                # Map labels to indices
                if data['label'] not in label_mapping:
                    label_mapping[data['label']] = current_index
                    current_index += 1
                labels.append(label_mapping[data['label']])
        
        return texts, labels, label_mapping

    # Process .csv file
    elif file_path.endswith('.csv'):
        # Read CSV, skip the header row
        df = pd.read_csv(file_path, header=0, names=['text', 'label'])
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        return texts, labels, None
    
    else:
        raise ValueError("Unsupported file format. Only .jsonl and .csv are supported.")

def preprocess_20_newsgroups(per_topic_samples = 100, shuffle=True, topics=None):
    newsgroups = datasets.fetch_20newsgroups(subset='all')
    group_by_topics = defaultdict(list)
    target_to_name = dict(enumerate(newsgroups["target_names"]))
    for target, text in zip(newsgroups['target'], newsgroups['data']):
        if topics is None or target_to_name[target] in topics:
            group_by_topics[target].append(text)
    if per_topic_samples is None:
        all_data = [(text, target) for target in group_by_topics for text in group_by_topics[target]]
    else:
        all_data = [(text, target) for target in group_by_topics for text in group_by_topics[target][:per_topic_samples]]
    if shuffle:
        random.shuffle(all_data)
    text, labels = zip(*all_data)
    return text, labels

def reorder_labels(label_array):
    label_to_new_label_mapping = {}
    for i, old_label in enumerate(list(set(label_array))):
        label_to_new_label_mapping[old_label] = i
    return [label_to_new_label_mapping[l] for l in label_array]

def sample_square_points(lower_left_corner, upper_right_corner, n_points=10, corner_offset=1.0, seed=0):
    x_boundaries = (lower_left_corner[0], upper_right_corner[0])
    y_boundaries = (lower_left_corner[1], upper_right_corner[1])
    
    corner_points = [(lower_left_corner[0] + corner_offset, lower_left_corner[1] + corner_offset),
                     (lower_left_corner[0] + corner_offset, upper_right_corner[1] - corner_offset),
                     (upper_right_corner[0] - corner_offset, upper_right_corner[1] - corner_offset),
                     (upper_right_corner[0] - corner_offset, lower_left_corner[1] + corner_offset),]
    '''

    corner_points = []
    '''

    np.random.seed(seed)
    x_samples = np.random.uniform(x_boundaries[0] + 0.00001, x_boundaries[1], size=n_points - 2)
    y_samples = np.random.uniform(y_boundaries[0] + 0.00001, y_boundaries[1], size=n_points - 2)

    sampled_points = corner_points + list(zip(x_samples, y_samples))
    sampled_points = [list(p) for p in sampled_points]
    return sampled_points

                

def generate_synthetic_data(n_samples_per_cluster, global_seed=2022):
    # 5 squares
    points = []
    labels = []
    square_1 = sample_square_points((0, 0), (5, 5), n_points=n_samples_per_cluster, seed=0)
    points.extend(square_1)
    labels.extend([0 for _ in square_1])
    square_2 = sample_square_points((4, 4), (8, 8), n_points=n_samples_per_cluster, seed=1)
    points.extend(square_2)
    labels.extend([1 for _ in square_2])
    square_3 = sample_square_points((7, 7), (10, 10), n_points=n_samples_per_cluster, seed=2)
    points.extend(square_3)
    labels.extend([2 for _ in square_3])
    square_4 = sample_square_points((7, 1), (11, 5), n_points=n_samples_per_cluster, seed=3)
    points.extend(square_4)
    labels.extend([3 for _ in square_4])
    square_5 = sample_square_points((2, 7), (5, 10), n_points=n_samples_per_cluster, seed=4)
    points.extend(square_5)
    labels.extend([4 for _ in square_5])
    combined_data = list(zip(points, labels))

    np.random.seed(global_seed)
    np.random.shuffle(combined_data)  
    points, labels = zip(*combined_data)
    return np.array(points), labels

def get_average_dse_embedding(texts):
    model = AutoModel.from_pretrained("aws-ai/dse-bert-base")
    tokenizer = AutoTokenizer.from_pretrained("aws-ai/dse-bert-base")
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Calculate the sentence embeddings by averaging the embeddings of non-padding words
    with torch.no_grad():
        embeddings = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = torch.sum(embeddings[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings.numpy()

def load_tweet(cache_path = None, encoder="instructor"):

    file_path = f"data/tweet.csv"

    result = read_data(file_path)
    if len(result) == 3:
        texts, labels, label_mapping = result
        print(f"Label Mapping: {label_mapping}")
    else:
        texts, labels = result
        print("Label Mapping: None")
    print(f"Texts: {texts[:2]}")  # Print first 2 texts
    print(f"Labels: {labels[:2]}")  # Print first 2 labels

    if encoder == "instructor":
        if cache_path is not None and os.path.exists(cache_path):
            # embeddings = pickle.load(open(cache_path, 'rb'))
            embeddings = torch.load(open(cache_path,'rb'))
            check_tensor_device(embeddings)

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()  
            embeddings = normalize_rows(embeddings)
            print("normalize_rows done")
        else:
            model = INSTRUCTOR('hkunlp/instructor-large')
            # prompt = "Represent utterances for intent classification: "
            prompt = "Represent news for type classification: "
            embeddings = model.encode([[prompt, text] for text in texts])
            if cache_path is not None:
                pickle.dump(embeddings, open(cache_path, 'wb'))
    elif encoder == "DSE":
        embeddings = get_average_dse_embedding(texts)
    else:
        raise NotImplementedError(f"Unexpected encoder {encoder} given")

    return embeddings, labels, list(texts)

def load_goemo(cache_path = None, encoder="instructor"):

    file_path = f"data/go_emo.jsonl"

    result = read_data(file_path)
    if len(result) == 3:
        texts, labels, label_mapping = result
        print(f"Label Mapping: {label_mapping}")
    else:
        texts, labels = result
        print("Label Mapping: None")

    if encoder == "instructor":
        if cache_path is not None and os.path.exists(cache_path):
            # embeddings = pickle.load(open(cache_path, 'rb'))
            embeddings = torch.load(open(cache_path,'rb'))
            check_tensor_device(embeddings)
        else:
            model = INSTRUCTOR('hkunlp/instructor-large')
            # prompt = "Represent utterances for intent classification: "
            prompt = "Represent news for type classification: "
            embeddings = model.encode([[prompt, text] for text in texts])
            if cache_path is not None:
                pickle.dump(embeddings, open(cache_path, 'wb'))
    elif encoder == "DSE":
        embeddings = get_average_dse_embedding(texts)
    else:
        raise NotImplementedError(f"Unexpected encoder {encoder} given")

    return embeddings, labels, list(texts)

def load_clinc(cache_path = None, encoder="instructor"):

    file_path = "data/clinic.jsonl"

    result = read_data(file_path)
    if len(result) == 3:
        texts, labels, label_mapping = result
        print(f"Label Mapping: {label_mapping}")
    else:
        texts, labels = result
        print("Label Mapping: None")
    print(f"Texts: {texts[:2]}")  # Print first 2 texts
    print(f"Labels: {labels[:2]}")  # Print first 2 labels

    if encoder == "instructor":
        if cache_path is not None and os.path.exists(cache_path):
            # embeddings = pickle.load(open(cache_path, 'rb'))
            embeddings = torch.load(open(cache_path,'rb'))
            check_tensor_device(embeddings)
        else:
            model = INSTRUCTOR('hkunlp/instructor-large')
            # prompt = "Represent utterances for intent classification: "
            prompt = "Represent utterances for domain classification: "
            embeddings = model.encode([[prompt, text] for text in texts])
            if cache_path is not None:
                pickle.dump(embeddings, open(cache_path, 'wb'))
    elif encoder == "DSE":
        embeddings = get_average_dse_embedding(texts)
    else:
        raise NotImplementedError(f"Unexpected encoder {encoder} given")

    return embeddings, labels, list(texts)

def load_bbc(cache_path = None, encoder="instructor"):

    file_path = f"data/bbc.csv"

    result = read_data(file_path)
    if len(result) == 3:
        texts, labels, label_mapping = result
        print(f"Label Mapping: {label_mapping}")
    else:
        texts, labels = result
        print("Label Mapping: None")
    print(f"Texts: {texts[:2]}")  # Print first 2 texts
    print(f"Labels: {labels[:2]}")  # Print first 2 labels

    if encoder == "instructor":
        if cache_path is not None and os.path.exists(cache_path):
            # embeddings = pickle.load(open(cache_path, 'rb'))
            embeddings = torch.load(open(cache_path,'rb'))
            check_tensor_device(embeddings)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()  
            embeddings = normalize_rows(embeddings)
            print("normalize_rows done")
        else:
            model = INSTRUCTOR('hkunlp/instructor-large')
            # prompt = "Represent utterances for intent classification: "
            prompt = "Represent news for type classification: "
            embeddings = model.encode([[prompt, text] for text in texts])
            if cache_path is not None:
                pickle.dump(embeddings, open(cache_path, 'wb'))
    elif encoder == "DSE":
        embeddings = get_average_dse_embedding(texts)
    else:
        raise NotImplementedError(f"Unexpected encoder {encoder} given")

    return embeddings, labels, list(texts)

def load_corpus(cache_path = None, file_path = None, encoder_name = "instructor"):

    result = read_data(file_path)
    if len(result) == 3:
        texts, labels, label_mapping = result
        print(f"Label Mapping: {label_mapping}")
    else:
        texts, labels = result
        print("Label Mapping: None")
    print(f"Texts: {texts[:2]}")  # Print first 2 texts
    print(f"Labels: {labels[:2]}")  # Print first 2 labels

    if cache_path is not None and os.path.exists(cache_path):
        embeddings = torch.load(open(cache_path,'rb'))
        check_tensor_device(embeddings)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()  

        if encoder_name == "instructor":
            embeddings = normalize_rows(embeddings)
            print("normalize_rows done")
        elif encoder_name == "sentencebert":
            embeddings = normalize(embeddings, norm='l2')
            print("normalize done")
        else:
            embeddings = normalize(embeddings, norm='l2')
            print("normalize done")

    else:
        raise NotImplementedError(f"Unexpected")

    return embeddings, labels, list(texts)

def load_bank77(cache_path = None, encoder="instructor"):
    dataset = load_dataset_hf("banking77")
    test_split = dataset["test"]
    texts = test_split["text"]
    labels = test_split["label"]

    if encoder == "instructor":
        if cache_path is not None and os.path.exists(cache_path):
            embeddings = pickle.load(open(cache_path, 'rb'))
        else:
            model = INSTRUCTOR('hkunlp/instructor-large')
            prompt = "Represent the bank purpose for classification: "
            embeddings = model.encode([[prompt, text] for text in texts])
            if cache_path is not None:
                pickle.dump(embeddings, open(cache_path, 'wb'))
    elif encoder == "DSE":
        embeddings = get_average_dse_embedding(texts)
    else:
        raise NotImplementedError(f"Unexpected encoder {encoder} given")

    return embeddings, labels, texts

def process_sentence_punctuation(sentences):
    processed_sentence_set = []
    for s in sentences:
        processed_sentence_set.append(s.replace("-LRB-", "(").replace("-RRB-", ")"))
    return processed_sentence_set

def load_dataset(dataset_name, data_path, dataset_split=None, use_dse_encoder=False):
    assert dataset_name in ["iris", "tweet", "clinc", "bank77", "20_newsgroups_all", "20_newsgroups_full", "20_newsgroups_sim3", "20_newsgroups_diff3", "reverb45k", "OPIEC59k", "reverb45k-raw", "OPIEC59k-raw", "OPIEC59k-kg", "OPIEC59k-text", "synthetic_data"]
    if dataset_name == "iris":
        samples, gold_cluster_ids = datasets.load_iris(return_X_y=True)
        documents = ["" for _ in samples]
        side_information = None
    if dataset_name == "tweet":
        if use_dse_encoder:
            samples, gold_cluster_ids, side_information = load_tweet(data_path, encoder="DSE")
        else:
            samples, gold_cluster_ids, side_information = load_tweet(data_path)
        documents = side_information
    if dataset_name == "clinc":
        if use_dse_encoder:
            samples, gold_cluster_ids, side_information = load_clinc(encoder="DSE")
        else:
            samples, gold_cluster_ids, side_information = load_clinc()
        documents = side_information
    if dataset_name == "bank77":
        if use_dse_encoder:
            samples, gold_cluster_ids, side_information = load_bank77(encoder="DSE")
        else:
            samples, gold_cluster_ids, side_information = load_bank77()
        documents = side_information
    elif dataset_name == "20_newsgroups_all":
        samples, gold_cluster_ids = preprocess_20_newsgroups(per_topic_samples=100)
        documents = samples
        side_information = None
    elif dataset_name == "20_newsgroups_full":
        samples, gold_cluster_ids = preprocess_20_newsgroups(per_topic_samples=None)
        documents = samples
        side_information = None
    elif dataset_name == "20_newsgroups_sim3":
        samples, gold_cluster_ids = preprocess_20_newsgroups(topics=["comp.graphics", "comp.os.ms-windows.misc", "comp.windows.x"])
        gold_cluster_ids = reorder_labels(gold_cluster_ids)
        documents = samples
        side_information = None
    elif dataset_name == "20_newsgroups_diff3":
        samples, gold_cluster_ids = preprocess_20_newsgroups(topics=["alt.atheism", "rec.sport.baseball", "sci.space"])
        gold_cluster_ids = reorder_labels(gold_cluster_ids)
        documents = samples
        side_information = None
    elif dataset_name == "synthetic_data":
        samples, gold_cluster_ids = generate_synthetic_data(n_samples_per_cluster=20)
        documents = ["" for _ in samples]
        side_information = None
    elif dataset_name.split('-')[0] == "OPIEC59k" or dataset_name.split('-')[0] == "reverb45k":
        name_constituents = dataset_name.split("-")
        if len(name_constituents) == 2 and name_constituents[1] in ["kg", "text"]:
            dataset_name = name_constituents[0]
            modality_type = name_constituents[1]
        elif len(name_constituents) == 2 and name_constituents[1] == "raw":
            dataset_name = name_constituents[0]
            modality_type = "all"
        else:
            modality_type = "all"

        # set up OPIEC59k evaluation set
        MockArgs = namedtuple("MockArgs", ["dataset", "file_triples", "file_entEmbed", "file_relEmbed", "file_entClust", "file_relClust", "file_sideinfo", "file_sideinfo_pkl", "file_results", "out_path", "data_path", "use_assume"])
        file_triples = '/triples.txt'  # Location for caching triples
        file_entEmbed = '/embed_ent.pkl'  # Location for caching learned embeddings for noun phrases
        file_relEmbed = '/embed_rel.pkl'  # Location for caching learned embeddings for relation phrases
        file_entClust = '/cluster_ent.txt'  # Location for caching Entity clustering results
        file_relClust = '/cluster_rel.txt'  # Location for caching Relation clustering results
        file_sideinfo = '/side_info.txt'  # Location for caching side information extracted for the KG (for display)
        file_sideinfo_pkl = '/side_info.pkl'  # Location for caching side information extracted for the KG (binary)
        file_results = '/results.json'  # Location for loading hyperparameters

        dataset_processed_version_name = dataset_name + '_' + dataset_split + '_' + '1'
        # This convoluted path operation goes from the path to the dataset's data directory to the path of the output files derived from data
        out_dir = os.path.join(os.path.abspath(os.path.join(data_path, os.pardir)), "output")
        out_path = os.path.join("/projects/ogma2/users/extra_storage", dataset_processed_version_name)
        dataset_file = os.path.join(data_path, dataset_name, dataset_name + '_' + dataset_split)
        use_assume = True
        mock_args = MockArgs(dataset_name, file_triples, file_entEmbed, file_relEmbed, file_entClust, file_relClust, file_sideinfo, file_sideinfo_pkl, file_results, out_path, dataset_file, use_assume)
        if dataset_name.split('-')[0] == "OPIEC59k":
            cmvc = CMVC_Main_opiec(mock_args)
        elif dataset_name.split('-')[0] == "reverb45k":
            cmvc = CMVC_Main_reverb(mock_args)

        print(f"Loaded triples")

        cmvc.get_sideInfo() 

        print(f"Loading sideInfo from CMVC")

        kg_features = np.load(open(os.path.join(data_path, dataset_name, f"{dataset_split}_relation_view_embed.npz"), 'rb'))
        bert_features = np.load(open(os.path.join(data_path, dataset_name, f"{dataset_split}_context_view_embed.npz"), 'rb'))
        cmvc.kg_dimension = kg_features.shape[1]
        cmvc.bert_dimension = bert_features.shape[1]
        if modality_type == "kg":
            samples = kg_features
        elif modality_type == "text":
            samples = bert_features
        elif modality_type == "all":
            samples = np.hstack([kg_features, bert_features])
        else:
            raise NotImplementedError

        ent_ids = [cmvc.side_info.ent2id[trp['triple'][0]] for trp in cmvc.side_info.triples]
        cluster_names = [list(cmvc.true_ent2clust[trp['triple_unique'][0]])[0] for trp in cmvc.side_info.triples]
        cluster_name_to_id = {}
        for c in cluster_names:
            if c not in cluster_name_to_id:
                cluster_name_to_id[c] = len(cluster_name_to_id)

        assert set(ent_ids) == set(list(range(max(ent_ids) + 1)))
        gold_cluster_ids = [None for i in range(max(ent_ids) + 1)]
        for i, ent_id in enumerate(ent_ids):
            cluster_name = cluster_names[i]
            gold_cluster_ids[ent_id] = cluster_name_to_id[cluster_name]


        '''
        entity_to_cluster_name = {}
        for i, trp in enumerate(cmvc.side_info.triples):
            ent_name = trp['triple'][0]
            cluster_name = cluster_names[i]
            entity_to_cluster_name[ent_name] = cluster_name
        '''

        cluster_id_to_name = invertDic(cluster_name_to_id, 'o2o')
        cmvc.cluster_id_to_name = cluster_id_to_name
        side_information = cmvc

        side_info = side_information.side_info
        cache_dir = "/projects/ogma1/few-shot-clustering/few_shot_clustering/file/gpt3_cache"
        sentence_unprocessing_mapping_file = os.path.join(cache_dir, f"{dataset_name}_test_sentence_unprocessing_map.json")
        sentence_unprocessing_mapping = json.load(open(sentence_unprocessing_mapping_file))
        selected_sentences = []
        ents = []
        for i in range(len(samples)):
            try:
                ents.append(side_info.id2ent[i])
            except:
                breakpoint()
            entity_sentence_idxs = side_info.ent_id2sentence_list[i]
            unprocessed_sentences = [sentence_unprocessing_mapping[side_info.sentence_List[j]] for j in entity_sentence_idxs]
            entity_sentences = process_sentence_punctuation(unprocessed_sentences)
            entity_sentences_dedup = list(set(entity_sentences))

            longest_sentences = sorted([s for s in entity_sentences_dedup if len(s) < 599], key=len)
            selected_sentences.append(list(set(longest_sentences[:3])))

        documents = []
        context_labels = ["a", "b", "c", "d"]
        for ent, sentences in zip(ents, selected_sentences):
            combined_sentences = "\n".join([context_labels[ind] + ") " + '"' + sent + '"' for ind, sent in enumerate(sentences)])
            context_text = f"""{ent}

Context Sentences:\n{combined_sentences}
"""
            documents.append(context_text)


    return samples, gold_cluster_ids, documents, side_information

import os
import json
import pandas as pd
import argparse
from baselines.test_bert import bert_uncased, sentence_bert
from baselines.test_gpt import embed_text
from baselines.test_llama import llama
from baselines.test_TF_IDF import TF_IDF
from baselines.test_instruct import test_instruct
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

# Example usage
if __name__ == "__main__":
    # Change this to the file you want to process
    parser = argparse.ArgumentParser(description="Cequel")
    parser.add_argument("--dataset", type=str, default="bank77")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model", type=str, default="bert")
    parser.add_argument("--output_dir", type=str, default="embeddings")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    file_path = os.path.join(args.data_dir, args.dataset)
    if args.dataset in ['bbc', 'tweet']:
        file_path = file_path + ".csv"
    elif args.dataset in ['bank77', 'clinic', 'massive_scenario', 'reddit']:
        file_path = file_path + ".jsonl"
    else:
        raise ValueError("Unsupported dataset. Only bbc, tweet, bank77, clinic, massive_scenario, and reddit are supported.")
    # file_path = "data/bbc.csv"
    try:
        result = read_data(file_path)
        if len(result) == 3:
            texts, labels, label_mapping = result
            print(f"Label Mapping: {label_mapping}")
        else:
            texts, labels = result
            print("Label Mapping: None")
    except Exception as e:
        print(f"Error: {e}")
    if args.model in ['bert','roberta', 'distilbert' ,'distilroberta']:
        bert_uncased(texts, args, args.model)
    elif args.model in ['sentence_bert']:
        sentence_bert(texts, args, args.model)
    elif args.model in ['gpt']:
        embed_text(texts, args)
    elif args.model in ['llama']:
        llama(texts, args)
    elif args.model in ['tfidf']:
        TF_IDF(texts, args)
    elif args.model in ['instructor']:
        test_instruct(texts, args)
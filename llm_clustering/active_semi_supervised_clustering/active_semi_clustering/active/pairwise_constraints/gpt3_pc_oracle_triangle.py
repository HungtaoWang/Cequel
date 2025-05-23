import json
import jsonlines
import math
import os
import time
import errno
import signal
import functools
import requests


import openai

from .example_oracle import MaximumQueriesExceeded

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

@timeout(5, os.strerror(errno.ETIMEDOUT))
def call_chatgpt(prompt, num_predictions, temperature=1.0, max_tokens=1, timeout=2.0):
    apiKey = ""  # Replace with your actual API Key
    basicUrl = ""
    modelName = ""
    apiVersion = ""

    conversation = [{"role": "user", "content": prompt}]
    url = f"{basicUrl}/deployments/{modelName}/chat/completions/?api-version={apiVersion}"
    headers = {'Content-Type': 'application/json', 'api-key': apiKey}
    payload = {
        'messages': conversation,
        'temperature': temperature,  # Use temperature parameter
        'max_tokens': max_tokens,  # Use max_tokens parameter
        'n': num_predictions  # Use num_predictions parameter
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)  # Use timeout parameter
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception during API call: {e}")
        return None

def construct_pairwise_oracle_single_example(document_i, document_j, document_k, label, dataset_name, prompt_suffix = None, text_type = None, add_label=True):
    if prompt_suffix is None:
        if dataset_name == "OPIEC59k":
            prompt_suffix = "link to the same entity's article on Wikipedia?"
        elif dataset_name == "reverb45k":
            prompt_suffix = "link to the same entity on a knowledge graph like Freebase?"
        elif dataset_name == "tweet":
            prompt_suffix = "discuss the same topic?"
        elif dataset_name == "clinc":
            # prompt_suffix = "express the same general intent?"
            # prompt_suffix = "correspond to the same general domain?"
            prompt_suffix = "which of these utterances belong to the same domain?"
        elif dataset_name == "bbc":
            # prompt_suffix = "express the same general intent?"
            # prompt_suffix = "correspond to the same general domain?"
            prompt_suffix = "which of these news belong to the same type?"
        elif dataset_name == "bank77":
            prompt_suffix = "express the same general intent?"
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Set a prompt_suffix")

    if text_type is None:
        if dataset_name == "OPIEC59k" or dataset_name == "reverb45k":
            text_type = "Entity"
        else:
            if dataset_name == "tweet":
                text_type = "Tweet"
            elif dataset_name == "clinc":
                text_type = "Utterance"
            elif dataset_name == "bank77":
                text_type = "Utterance"
            else:
                raise ValueError(f"Dataset {dataset_name} not supported.")
    template_prefix = f"""{text_type} #1: {document_i}
{text_type} #2: {document_j}
{text_type} #3: {document_k}

Given this context, do {text_type.lower()} #1, {text_type.lower()} #2 and {text_type.lower()} #3 likely {prompt_suffix} 

a) All are Same Domain  
b) ONLY #1 and #2 are Same Domain 
c) ONLY #1 and #3 are Same Domain  
d) ONLY #2 and #3 are Same Domain 
e) NONE """

    if add_label:
        full_example = template_prefix + label
    else:
        full_example = template_prefix
    return full_example

class GPT3Oracle_triangle:
    def __init__(self, X, prompt, documents, dataset_name=None, prompt_suffix=None, text_type=None, cache_file=None, max_queries_cnt=2500, num_predictions=5, read_only=False):
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt
        self.num_predictions = num_predictions
        self.documents = documents
        self.dataset_name = dataset_name
        self.prompt_suffix = prompt_suffix
        self.text_type = text_type

        self.cache_file = cache_file
        if os.path.exists(self.cache_file):
            self.cache_rows = list(jsonlines.open(self.cache_file))
        else:
            self.cache_rows = []
        if not read_only:
            self.cache_writer = jsonlines.open(self.cache_file, mode='a', flush=True)
        else:
            self.cache_writer = jsonlines.open(self.cache_file, mode='r')

        # self.NUM_RETRIES = 2
        self.NUM_RETRIES = 5
        self.read_only = read_only

        self.gpt3_pairwise_labels = {}

        for row in self.cache_rows:
            sorted_pair_list = sorted([row["context1"], row["context2"], row["context3"]])
            self.gpt3_pairwise_labels[tuple(sorted_pair_list)] = row["labels"]

        self.prompt = prompt

    @staticmethod
    def filter_high_entropy_predictions_multi_class(pair_labels):
        """
        Determine the label that appears most frequently in the input.

        Parameters:
        - pair_labels: A list of predicted labels (e.g., ['a', 'b', 'a', 'c', 'a']).

        Returns:
        - The label that appears most frequently ('a', 'b', 'c', 'd', 'e').
        - None if the input list is empty.
        """
        if not pair_labels:
            return None  # Handle empty input case

        # Ensure all labels are valid
        assert all(label in ['a', 'b', 'c', 'd', 'e'] for label in pair_labels), "Invalid labels in input."

        # Count occurrences of each label and find the most frequent one
        label_counts = {label: pair_labels.count(label) for label in ['a', 'b', 'c', 'd', 'e']}
        most_frequent_label = max(label_counts, key=label_counts.get)

        return most_frequent_label


    def query(self, i, j, k):
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            try:
                sorted_pair_list = sorted([self.documents[i], self.documents[j], self.documents[k]])
            except:
                breakpoint()

            sorted_pair = tuple(sorted_pair_list)

            if  sorted_pair in self.gpt3_pairwise_labels:
                return self.filter_high_entropy_predictions_multi_class(self.gpt3_pairwise_labels[sorted_pair])

            prompt_prefix = self.prompt
            context1 = self.documents[i]
            context2 = self.documents[j]
            context3 = self.documents[k]
            prompt_to_be_completed = construct_pairwise_oracle_single_example(self.documents[i], self.documents[j], self.documents[k], label=None, add_label=False, dataset_name=self.dataset_name, prompt_suffix=self.prompt_suffix, text_type=self.text_type)
            prompt = prompt_prefix + "\n\n" + prompt_to_be_completed

            print("PROMPT:\n" + prompt)

            pair_labels_not_none = []

            failure = True
            num_retries = 0
            while failure and num_retries < self.NUM_RETRIES:
                cache_row = None
                try:
                    start = time.perf_counter()
                    response = call_chatgpt(prompt, self.num_predictions, temperature=1.0, max_tokens=1, timeout=2.0)
                    print(f"response took {round(time.perf_counter()-start, 2)} seconds")

                    pair_labels = []
                    for choice in response["choices"]:
                        message = choice["message"]["content"]  
                        if message.strip() == "a":
                            pair_label = "a"
                        elif message.strip() == "b":
                            pair_label = "b"
                        elif message.strip() == "c":
                            pair_label = "c"
                        elif message.strip() == "d":
                            pair_label = "d"
                        elif message.strip() == "e":
                            pair_label = "e"
                        else:
                            pair_label = None
                        pair_labels.append(pair_label)

                    print(f"labels:\n{pair_labels}\n\n")
                    pair_labels_not_none = [x for x in pair_labels if x is not None]
                    if len(pair_labels_not_none) <= self.num_predictions / 2:
                        time.sleep(0.2)
                    else:
                        cache_row = {
                                     "labels": pair_labels_not_none,
                                    #  "p_true": round(sum(pair_labels_not_none) / len(pair_labels_not_none), 4),
                                     "context1": context1,
                                     "context2": context2,
                                     "context3": context3
                                     }
                        self.cache_writer.write(cache_row)
                        self.gpt3_pairwise_labels[sorted_pair] = pair_labels_not_none
                        failure = False


                    num_retries += 1
                    end = time.perf_counter()
                    if end - start < 0.3:
                        time.sleep(0.3 - (end - start))
                except Exception as e:
                    print(e)
                    time.sleep(3)

            if failure:
                return None
            else:
                return self.filter_high_entropy_predictions_multi_class(pair_labels_not_none)
        else:
            raise MaximumQueriesExceeded

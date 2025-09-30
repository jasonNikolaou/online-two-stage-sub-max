import json
import numpy as np
import pickle
import sys
import os
import random
import gzip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WTP import Potential, WTP

np.random.seed(seed = 42)

def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def build_query_support_indices(queries, title_to_index):
    """
    For each query, map its supporting document titles (from metadata)
    to integer indices using the title_to_index map.
    """
    support_doc_ids_per_query = []

    for query in queries:
        try:
            titles = query['metadata']['supporting_facts']
            doc_ids = [title_to_index[title[0]] for title in titles if title[0] in title_to_index]
        except Exception as e:
            print(f"Error processing query {query.get('query_id', '[unknown]')}: {e}")
            doc_ids = []
        support_doc_ids_per_query.append(doc_ids)

    return support_doc_ids_per_query

def write_dataset(T, queries, corpus, num_funcs=None):

    if T is None: T = len(queries)

    if num_funcs is None:
        num_funcs = T
    
    working_queries = queries[:T]
    all_article_titles = set()

    for query in working_queries:
        metadata = query['metadata']
        if metadata is not None:
            titles = metadata['supporting_facts']
            for title in titles:
                all_article_titles.add(title[0])
    
    n = len(all_article_titles)

    print(f'n = {n}')

    title_to_index = {doc['title']: idx for idx, doc in enumerate(corpus) if doc['title'] in all_article_titles}
    title_to_id = {}

    index = 0
    for title in all_article_titles:
        title_to_id[title] = index
        index += 1
    

    functions = build_query_support_indices(working_queries, title_to_id)

    wtps = []
    for ids in functions:
        if len(ids) == 0:
            continue
        w = np.zeros(n)
        for id in ids:
            w[id] = 1
        
        potential = Potential(1,w)
        wtp = WTP([potential], [1])
        wtps.append(wtp)
    
    random_indices = [
        random.randint(0, 30) if random.random() < 0.7 else random.randint(31, len(wtps)-1)
        for _ in range(num_funcs)
    ]

    random_functions = [wtps[i] for i in random_indices]
    
    print(f'Number of functions : {len(random_functions)}')


    file = f'../instances/hotpotqa.pkl'
    with gzip.open(file, 'wb') as f:
        pickle.dump(random_functions, f)
    print(f'List saved to {file}')


def main():
    corpus_path = "corpus.jsonl"
    queries_path = "queries.jsonl"

    print("Loading corpus...")
    corpus = load_jsonl(corpus_path)
    print(f"Loaded {len(corpus)} documents.")

    print("Loading queries...")
    queries = load_jsonl(queries_path)
    print(f"Loaded {len(queries)} queries.")

    write_dataset(None, queries, corpus, num_funcs=500)



if __name__ == "__main__":
    main()
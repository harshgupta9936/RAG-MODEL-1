import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-text",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding


# jsons = os.listdir("jsons")  # List all the jsons for old jsons
jsons = os.listdir("new_jsons")  # List all the jsons for new jsons ( reduced chunks )
my_dicts = []
chunk_id = 0

for json_file in jsons:
    # with open(f"jsons/{json_file}") as f:        # old jsons
    with open(f"new_jsons/{json_file}") as f:       # new jsons
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
# print(df)

#save this data frame:
joblib.dump(df, 'embeddings.joblib')
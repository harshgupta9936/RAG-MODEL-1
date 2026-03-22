import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from openai import OpenAI
from apikey import my_api_key

client = OpenAI(api_key=my_api_key)

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-text",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "deepseek-r1",
        # "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response

# def inference_openai(prompt):
#     print ("thinking...")
#     response = client.responses.create(
#     model="gpt-5.4",
#     input=prompt
#     )
#     return response.output_text


df= joblib.load('embeddings.joblib')

input_query = input("Ask a question: ")
question_embedding = create_embedding([input_query])[0]
# print (question_embedding)

# print(np.vstack(df['embedding'].values))
# print(df['embedding'].shape)
# find similarity of question embeddings with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[["title","number", "text"]])

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])


prompt = f'''In these Videos there are skills taught for Data Analyst Jobs. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{input_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open("prompt.txt", "w") as f:
    f.write(prompt)

# for ollama mmodel
response = inference(prompt)["response"]
print(response)

#for gpt model:
# response = inference_openai(prompt)


with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)
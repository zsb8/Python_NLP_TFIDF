import numpy as np
import json
import nltk
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("stsb-roberta-large")
sentence_embeddings = []

with open("../fbtext.json", "r") as f:
    inputs = json.load(f)

count = 0
for row in inputs:
    count += 1
    if count % 100 == 0:
        print(f"processed {count} posts")
    sentences = nltk.sent_tokenize(row)
    embeddings = model.encode(sentences)
    sentence_embeddings.extend(list(zip(sentences, embeddings.tolist())))


with open("fb_sbert_embeddings.json", "w") as f:
    json.dump(sentence_embeddings, f, indent=4)

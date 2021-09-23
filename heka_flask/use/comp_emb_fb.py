import tensorflow as tf
import tensorflow_hub as hub
import nltk
import numpy as np
import json

sentence_embeddings = dict()

embed = hub.load("./models/use-large")
with open("../fbtext.json", "r") as f:
    inputs = json.load(f)

count = 0
for row in inputs:

    sentences = nltk.sent_tokenize(row)

    embeddings = embed(sentences).numpy()
    for i, emb in enumerate(embeddings):
        count += 1
        if count % 100 == 0:
            print(f"processed {count} sentences")
        sentence_embeddings[sentences[i]] = emb.tolist()

with open("fb_USE_embeddings.json", "w") as f:
    json.dump(sentence_embeddings, f, indent=4)

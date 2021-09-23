import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

physio_embeddings = dict()

embed = hub.load("./models/use")
with open("../physios_renamed.json", "r") as f:
    physios = json.load(f)

count = 0
for name, physio in physios.items():
    count += 1
    if count % 100 == 0:
        print(f"processed {count} physios")
    try:
        embeddings = embed(physio["reasons"])
    except KeyError:
        physio_embeddings[name] = []
        continue
    physio_embeddings[name] = [emb.tolist() for emb in embeddings.numpy()]

with open("use/physio_embeddings.json", "w") as f:
    json.dump(physio_embeddings, f, indent=4)

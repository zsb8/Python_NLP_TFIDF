from flask_restful import Resource, reqparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

import json
from time import time
from math import sqrt

model = SentenceTransformer("stsb-distilbert-base")
parser = reqparse.RequestParser()
parser.add_argument("input")

# Api testing
with open("physios_renamed.json", "r") as f:
    physios = json.load(f)

with open("sbert/physio_embeddings.json", "r") as f:
    physio_embeddings = json.load(f)


class MatchSBERT(Resource):
    def post(self):
        start = time()
        args = parser.parse_args()
        print(args)
        if not args["input"]:
            return ({}, 400)
        text_input = args["input"]
        temp = time()
        input_embedding = model.encode(text_input)
        print(f"embedding: {(temp-start)*1000}ms")
        cos_sim_list = []
        temp = time()
        for name, embeddings in physio_embeddings.items():
            if embeddings:
                # loopstart = time()

                physio_cos_sim_list = cosine_similarity([input_embedding], embeddings)[
                    0
                ]
                # print(f"sim list: {(time()-loopstart)*1000}ms")
                # loopstart = time()
                cos_sim_list.append((name, max(physio_cos_sim_list)))
                # print(f"max: {(time()-loopstart)*1000}ms")
                # break
            else:
                cos_sim_list.append((name, 0))

        print(f"cosine similarity: {(time()-temp)*1000}ms")
        temp = time()

        sorted_physio = sorted(cos_sim_list, reverse=True, key=lambda t: t[1])
        print(sorted_physio)
        print(f"sorting: {(time()-temp)*1000}ms")

        return (
            {
                "input": args["input"],
                "best_matches": {
                    physio[0]: physios[physio[0]] for physio in sorted_physio[:50]
                },
            },
            200,
        )


from flask_restful import Resource, reqparse
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("stsb-roberta-large")

parser = reqparse.RequestParser()
parser.add_argument("input")

# Api testing

with open("./sbert/fb_sbert_embeddings.json", "r") as f:
    fb_embeddings = json.load(f)


class SuggestSBERT(Resource):
    def post(self):
        args = parser.parse_args()
        if not args["input"]:
            return ({}, 400)
        text_input = args["input"]

        input_embedding = model.encode(text_input)

        cos_sim_list = []

        for sentence, embedding in fb_embeddings:
            cos_sim = util.pytorch_cos_sim(input_embedding, embedding).tolist()[0][0]
            cos_sim_list.append([sentence, cos_sim])

        # input_embedding = embed([args["text_input"]])
        # for name, embeddings in physio_embeddings.items():
        #     if embeddings:
        #         physio_inner_max.append(
        #             (name, max(np.inner(input_embedding, embeddings)[0]))
        #         )
        #     else:
        #         physio_inner_max.append((name, 0))

        # sorted_physio = sorted(physio_inner_max, reverse=True, key=lambda t: t[1])
        sorted_cos_sim = sorted(cos_sim_list, reverse=True, key=lambda l: l[1])

        return ({"input": args["input"], "best_matches": sorted_cos_sim[0:10]}, 200)


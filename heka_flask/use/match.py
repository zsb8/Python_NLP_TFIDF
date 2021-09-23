from flask_restful import Resource, reqparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

parser = reqparse.RequestParser()
parser.add_argument("text_input")
embed = hub.load("USE/models/use")

# Api testing
with open("physios_renamed.json", "r") as f:
    physios = json.load(f)

with open("USE/physio_embeddings.json", "r") as f:
    physio_embeddings = json.load(f)


class MatchUSE(Resource):
    def post(self):
        args = parser.parse_args()
        if not args["text_input"]:
            return ({}, 400)

        physio_inner_max = []

        input_embedding = embed([args["text_input"]])
        for name, embeddings in physio_embeddings.items():
            if embeddings:
                physio_inner_max.append(
                    (name, max(np.inner(input_embedding, embeddings)[0]))
                )
            else:
                physio_inner_max.append((name, 0))

        sorted_physio = sorted(physio_inner_max, reverse=True, key=lambda t: t[1])

        return (
            {
                "input": args["text_input"],
                "best_matches": {
                    physio[0]: physios[physio[0]] for physio in sorted_physio[:50]
                },
            },
            200,
        )


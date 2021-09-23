from flask_restful import Resource, reqparse
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import Heka_utils as utils
import time

def find_name(my_id):
    sql = """select a."firstName"||' '||"lastName" as name from "Clients" as a inner join  "Physios" as b """ \
           f"""on a.id=b."ClientId" where a."firstName" is not null and a."lastName" is not null and b.id={my_id}    """
    result = utils.execute_sql(sql)
    if len(result)==0:
        return "NoName"
    else:
        return result[0][0]


with open("sbert/physio_embeddings.json", "r") as f:
    physio_embeddings = json.load(f)
    f.close()
model = SentenceTransformer("stsb-distilbert-base")
physio = dict()
parser = reqparse.RequestParser()
parser.add_argument("input")


class MatchSBERT(Resource):
    def post(self):
        args = parser.parse_args()
        if not args["input"]:
            return {}, 400
        text_input = args["input"]
        begin_time = int(time.time())
        input_embedding = model.encode(text_input)
        cos_sim_list = []
        for id, embeddings in physio_embeddings.items():
            if embeddings:
                physio_cos_sim_list = cosine_similarity([input_embedding], embeddings)[0]
                cos_sim_list.append((id, max(physio_cos_sim_list)))
        sorted_physio = sorted(cos_sim_list, reverse=True, key=lambda t: t[1])
        physio["input"] = text_input
        sub_dict = dict()
        for search_result in sorted_physio[:20]:
            physio_id = search_result[0]
            sub_dict[physio_id] = {"name": find_name(physio_id)}
        physio["best_matches"] = sub_dict
        end_time = int(time.time())
        print(f"This task spend 时间 {round((end_time - begin_time), 2)} seconds. stably only 8 seconds")
        return (
            physio,
            200,
        )
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_restful import Resource, reqparse
import time
import pandas as pd
import redis
import pickle
parser = reqparse.RequestParser()
parser.add_argument("input")
# get emb big data from redis, is a dict, cost 6 seconds
rs = redis.StrictRedis(host='192.168.50.1111')
emb_physio = rs.hgetall("emb_physio")
df = pd.DataFrame([emb_physio]).T
df.index = df.index.astype('int')
df.columns = ["emb"]
physio = dict()
class MatchSBERT(Resource):
    def post(self):
        args = parser.parse_args()
        if not args["input"]:
            return {}, 400
        begin_time = int(time.time())
        # calculate most fetch, cost 8 seconds
        model = SentenceTransformer("stsb-distilbert-base")
        text_input = args["input"]
        input_embedding = model.encode(text_input)
        df_temp = df.copy
        df_temp["emb"] = df_temp["emb"].apply(lambda x: max(cosine_similarity([input_embedding], pickle.loads(x))[0]))
        # make a dict including input content and sub_dict which including id and name,cost 0 seconds
        df2 = df_temp.sort_values(by='emb', ascending=False).head(20)
        physio["input"] = text_input
        rs2 = redis.StrictRedis(host='192.168.50.1111', decode_responses=True)
        physio_name_list = rs2.hmget("physio_name", df2.index)
        sub_dict = dict(zip(df2.index, physio_name_list))
        physio["best_matches"] = sub_dict
        end_time = int(time.time())
        print(f"This task spend  {round((end_time - begin_time), 2)} seconds. about 7 seconds")
        return (
            physio,
            200,
        )
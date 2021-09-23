from flask import Flask, request
from flask_restful import Resource, Api

# from use.match import MatchUSE
from sbert.match_db import MatchSBERT
#from sbert.match_db_redis import MatchSBERT
# from sbert.suggest_sbert import SuggestSBERT
from tfid import TFID

app = Flask(__name__)
api = Api(app)


class Root(Resource):
    def get(self):
        return "Heka's API"


api.add_resource(Root, "/")
# api.add_resource(MatchUSE, "/match/use")
api.add_resource(MatchSBERT, "/match/sbert")
# api.add_resource(SuggestSBERT, "/suggest/sbert")
api.add_resource(TFID, "/tfid")

if __name__ == "__main__":
    app.run()

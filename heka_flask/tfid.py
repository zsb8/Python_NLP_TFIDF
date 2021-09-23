from flask_restful import Resource, reqparse
from Heka_searchbar_NLP import result_sentence

parser = reqparse.RequestParser()
parser.add_argument("input")


class TFID(Resource):
    def post(self):
        args = parser.parse_args()
        data_from_front = args["input"]
        # 'long' mode is full sentence, 'short' mode is segment, 7 is words of showing one row
        result = result_sentence(data_from_front, "short", 7)
        return {"output": result}, 201


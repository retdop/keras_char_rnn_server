
from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask_restful import reqparse
from keras.models import load_model

import sample
from utils import TextLoader

txt = TextLoader()
model = load_model('python/python_256_0.001_256.h5')

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('start_text', type=str, help='Starting text')
parser.add_argument('n', type=int, help='Number of suggestions per request')

class Home(Resource):
    def get(self):
        return {'message': 'Have Fun!'}

class Generate(Resource):
    def get(self):
        args = parser.parse_args()
        sampled_text = sample.sample_chars(prime = args.start_text, n_chars = 100, diversity = 0.5, txt = txt, model = model)[len(args.start_text)::]
        result = {'completions': [sampled_text], 'start_text': args.start_text, 'time': 15}
        return result


api.add_resource(Home, '/')
api.add_resource(Generate, '/generate')

if __name__ == '__main__':
    app.run(port = 8080)

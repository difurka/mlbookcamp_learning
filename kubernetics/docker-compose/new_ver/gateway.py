
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

import os

from proto import np_to_protobuf


app = Flask('gateway')

host = os.getenv('TF_SERVING_HOST', 'localhost:8500')
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'clothing-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(X))
    return pb_request




def predict(url):
    # url = 'http://bit.ly/mlbookcamp-pants'

    preprocessor = create_preprocessor('xception', target_size=(299, 299))
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    result = prepare_response(pb_response)
    return result


def prepare_response(pb_response):
    preds = pb_response.outputs['dense_7'].float_val

    classes = [
        'dress',
        'hat',
        'longsleeve',
        'outwear',
        'pants',
        'shirt',
        'shoes',
        'shorts',
        'skirt',
        't-shirt'
    ]
    return dict(zip(classes, preds))

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    response = predict(url)
    return jsonify(response)


if __name__ == '__main__':
    # url = 'http://bit.ly/mlbookcamp-pants'
    # response = predict(url)
    # print(response)
    app.run(debug=True, host='0.0.0.0', port=9696)
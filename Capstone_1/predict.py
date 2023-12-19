
from flask import Flask, jsonify, request
import numpy as np
url = 'http://localhost:9696/predict'
import torch

import json
from src.model import SegNet
from src.utils import get_datasets, train, score_model, iou_pytorch

output_file = 'best_model.pth'


app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():

    checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
    model_seg = SegNet()
    # model_seg.load_state_dict(checkpoint)
    
    image_test = request.get_json()
    X = image_test['image']
    Y = image_test['lesion']
    image = torch.tensor(json.loads(X))
    lesion = torch.tensor(json.loads(Y))

   
    # image_test = [(image, lesion)]
    # score1 = score_model(model_seg, iou_pytorch, image_test, treshold=0.5)
    image = image.reshape(-1,264,264,3)
    print(image.shape)
    model_seg.eval()
    score1 = 0
    # print(type(data))
    # with torch.no_grad():
    # # # # #   for X_batch, Y_label in data:
    # # # #     #   print(type(data))
    # # # #     #   print(type(data))
    # # #     #   torch.sigmoid(
    #            model_seg(torch.tensor(image))
    #     #   print(Y_pred)
    # #     #   Y_pred = torch.where(Y_pred > 0.5, 1, 0)
    # #     #   score1 = iou_pytorch(Y_pred, lesion).mean().item()


    print(score1)
    score = 0.999
    result = {
        'score': float(score),
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

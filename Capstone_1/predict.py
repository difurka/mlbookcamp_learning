
import numpy as np
from flask import Flask, jsonify, request


import json

import torch
from src.model import SegNet
from src.utils import iou_pytorch

best_model_file = 'best_model.pth'
url = 'http://localhost:9696/predict'

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    checkpoint = torch.load(best_model_file, map_location=torch.device('cpu'))
    model_seg = SegNet()
    model_seg.load_state_dict(checkpoint)
    
    image_test = request.get_json()
    X = image_test['image']
    Y = image_test['lesion']
    image = np.array([np.array((json.loads(X)))], np.float32)
    lesion = np.array([np.array([np.array(json.loads(Y))], np.float32)])

    image = np.rollaxis(image, 3, 1)
    model_seg.eval()
    with torch.no_grad():
        Y_pred = torch.sigmoid(model_seg(torch.tensor(image)))
        Y_pred = torch.where(Y_pred > 0.5, 1, 0)
        score = iou_pytorch(Y_pred, torch.tensor(lesion)).mean().item()

    result = {
        'score': float(score),
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)


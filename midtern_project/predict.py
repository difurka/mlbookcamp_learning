
from flask import Flask, jsonify, request

url = 'http://localhost:9696/predict'
import pickle

import pandas as pd
import xgboost as xgb

output_file = 'model_tree.bin'

def prepare_train():
    X_train = pd.read_csv('./data/train.csv')
    X_train = X_train.drop_duplicates()
    X_train['TotalSpent'] = pd.to_numeric(X_train['TotalSpent'], errors='coerce')
    X_train['TotalSpent'] = X_train['TotalSpent'].fillna(X_train['TotalSpent'].mean())
    y_train = X_train.Churn

    del X_train['Churn']
    return X_train, y_train


with open(output_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


def predict_single(x_example, dv, model):
    x_ex_dict = dv.transform([x_example])
    dx = xgb.DMatrix(x_ex_dict )
    return model.predict(dx)


app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn),
    }

    
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
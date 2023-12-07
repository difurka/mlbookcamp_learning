FROM tensorflow/serving:2.7.0

COPY cl-model /models/clothing-model/1

ENV MODEL_NAME="clothing-model"
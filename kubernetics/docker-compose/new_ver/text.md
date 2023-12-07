saved_model_cli show --dir cl-model --tag_set serve --signature_def serving_default


docker run -it --rm \
  -p 8500:8500 \
  -v $(pwd)/cl-model:/models/clothing-model/1 \
  -e MODEL_NAME="clothing-model" \
  tensorflow/serving:2.7.0

///////////////

docker build -t zoomcamp_10_model:xception_v4_001 -f image_model.dockerfile .

docker run -it --rm  -p 8500:8500 zoomcamp_10_model:xception_v4_001

////////////////

docker build -t zoomcamp_10_gateway:001 -f image_gateway.dockerfile .

docker run -it --rm  -p 9696:9696 zoomcamp_10_gateway:001

///////////////

// start in detach
docker-compose up -d
//stop
docker-compose down
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

///////////////
docker build -t ping:001 .
docker run -it --rm  -p 9696:9696 ping:001

curl localhost:9696/ping

/////////

kubectl apply -f deployment.yaml 

kubectl get deployment
kubectl get pod


kubectl describe pod <> | less

kind load docker-image <im_name>

kubectl port-forward <pod-name> 9696:9696
curl localhost:9696/ping
/// service
kubectl apply -f service.yaml 
kubectl get service
kubectl port-forward service/ping 8080:80
curl localhost:8080/ping


///////////////
for model

kubectl apply -f model-deployment.yaml 
kind load docker-image zoomcamp_10_model:xception_v4_001
kubectl port-forward <> 8500:8500
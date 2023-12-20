prediction:
	gunicorn --bind 0.0.0.0:9696 predict:app

docker:
	docker run -it -p 9696:9696 seg-prediction:latest

create_image:
	docker build -t seg-prediction .

get_response:
	python3 get_response.py

train:
	python3 train.py

pred:
	python3 predict.py
prediction:
	gunicorn --bind 0.0.0.0:9696 predict:app

docker:
	docker run -it -p 9696:9696 churn-prediction:latest

create_image:
	docker build -t churn-prediction .

get_response:
	python3 get_response.py
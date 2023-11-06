# Midterm project

## Find a dataset

This competition is homework for module 3 of the advanced DLS stream (spring 2021). You will learn how to model customer churn for a telecom company. This task is very important in practice and algorithms for solving it are used in real telecom companies, because if we know that a client is going to leave us, then we will try to keep him by offering some kind of bonuses.(https://www.kaggle.com/competitions/advanced-dls-spring-2021/overview)


## Content of project

In file [mlzoomcamp_midterm_project.ipynb](mlzoomcamp_midterm_project.ipynb) the data is explored and prepared, different models are trained: Logistic Regression, DecisionTreeClassifier, Random Forest Classifier, and Gradient boosting, and CatBoost. Random Forest Classifier has given the best result for test dataset on Kaggle.

- In folder "data" there is dataset for training model. 
- In file [train.py](train.py) model for Random Forest Classifier(the best model) is trained.
- In file [predict.py](predict.py) prediction is created.
- In file [predict-test.py](predict-test.py) put the model into a web service
- In file [Dockerfile](Dockerfile) deploy it locally with Docker.
- Files "Pipfile" and "Pipfile.loc" are for environment.
- In file [makefile](makefile) some commands is located for create an image, and a container

So the next steps was done:
- Find a dataset (https://www.kaggle.com/competitions/advanced-dls-spring-2021/overview)
- Explore and prepare the data [mlzoomcamp_midterm_project.ipynb](mlzoomcamp_midterm_project.ipynb)
- Train the best model [mlzoomcamp_midterm_project.ipynb](mlzoomcamp_midterm_project.ipynb)
- Export the notebook into a script [train.py](train.py), [predict.py](predict.py)
- Put your model into a web service [predict.py](predict.py)
- Deploy it locally with Docker [Dockerfile](Dockerfile)


With any questions you can write to me in telegram @difurka
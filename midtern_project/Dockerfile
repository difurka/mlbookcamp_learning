# First install the python 3.8, the slim version uses less space
FROM python:3.10

# Install pipenv library in Docker 
RUN pip install pipenv

# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
# RUN pipenv requirements > requirements.txt && pip install -r requirements.txt

# Copy any python files and the model we had to the working directory of Docker 
COPY ["predict.py", "model_tree.bin", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our churn app to be running
# ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
ENTRYPOINT [ "bash" ]
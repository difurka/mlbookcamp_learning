# First install the python 3.8, the slim version uses less space
FROM ubuntu

# Install pipenv library in Docker 
RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.6 python3-distutils python3-pip python3-apt
# RUN apt update && apt upgrade
# RUN apt install python3.10
RUN apt install python3-pip -y
RUN pip install --upgrade pip
RUN pip install pipenv 


# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv requirements > requirements.txt && pip install -r requirements.txt

# Copy any python files and the model we had to the working directory of Docker 
COPY ["predict.py", "best_model.pth", "./"]
ADD src ./

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]

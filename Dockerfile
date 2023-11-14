FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# install system dependencies
RUN apt-get update && apt-get install nano

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
 
# install dependencies
RUN pip install --upgrade pip
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# copy project
#COPY . .


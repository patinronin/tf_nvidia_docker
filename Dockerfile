FROM tensorflow/tensorflow:latest-gpu
#FROM tensorflow/tensorflow:2.0.0-gpu-py3
ARG gpus=all

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY requirements.txt requirements.txt


RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . /app
EXPOSE 8000
CMD ["python", "./tensorflow_gpu_test/main.py"]

#  sudo docker build --tag tf_test .
# sudo docker run --runtime=nvidia --gpus all tf_test
# dont work the gpu in tensorflow 2.12
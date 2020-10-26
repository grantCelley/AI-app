From nvidia/cuda:10.2-devel-ubuntu18.04

ARG PYTHON_VERSION:3.8

COPY . /APP

WORKDIR /APP

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install torch torchvision \
  detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html

#CMD python3 /App/train/train.py

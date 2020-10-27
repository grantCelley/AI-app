FROM tensorflow/tensorflow:2.3.1-gpu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget


RUN git clone --depth 1 https://github.com/tensorflow/models

WORKDIR models/research
RUN protoc object_detection/protos/*.proto --python_out=. && \
  cp object_detection/packages/tf2/setup.py . && \
  python3 -m pip install -q . && \
  wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz && \
  tar -xf ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz && \
  mv ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint object_detection/test_data/

WORKDIR /app


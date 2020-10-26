FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget


RUN git clone --depth 1 https://github.com/tensorflow/models && \
  protoc models/research/object_detection/protos/*.proto --pyhton_out=models/research && \
  cp models/research/object_detection/packages/tf2/setup.py models/research && \
  python3 -m pip install -q models/research


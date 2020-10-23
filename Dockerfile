From pytorch/pytorch:latest

COPY . /App

WORKDIR /App

RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html

CMD python3 ./train/train.py

FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV DATASET_PATH=/data

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

RUN pip install --upgrade pip && pip install tqdm opencv-python

ENTRYPOINT [ "python" , "scripts/03_display.py" ]
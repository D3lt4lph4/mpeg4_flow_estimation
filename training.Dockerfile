FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV EXPERIMENTS_OUTPUT_DIRECTORY=/experiments
ENV DATASET_PATH=/data

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

RUN pip install --upgrade pip && pip install tqdm opencv-python scikit-learn matplotlib

ENTRYPOINT [ "python" , "scripts/01_training.py" ]
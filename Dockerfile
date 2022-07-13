FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV EXPERIMENTS_OUTPUT_DIRECTORY=/experiments
ENV DATASET_PATH=/data

WORKDIR /app

# NVIDIA bug https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

RUN pip install --upgrade pip && pip install tqdm opencv-python scikit-learn matplotlib

ENTRYPOINT [ "python" ]
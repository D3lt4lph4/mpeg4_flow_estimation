FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV EXPERIMENTS_OUTPUT_DIRECTORY=/experiments
ENV DATASET_PATH=/data

WORKDIR /app

# COPY requirements.txt .

# RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install --upgrade pip && pip install tqdm

ENTRYPOINT [ "python" , "scripts/01_training.py" ]
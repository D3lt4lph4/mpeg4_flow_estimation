# mpeg4_flow_estimation

## Setup the environment

A dockerfile is provided to simplify the deployment.
To build it, run the following command:

```bash
docker image build -t ${USER}/flow_estimation .
```

Alternatively, you can install the required dependecies and setup a virtual environment:

```bash
apt-get update
apt-get install -y ffmpeg libsm6 libxext6

python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip
pip install tqdm opencv-python scikit-learn matplotlib
```

In such case, simply run the commands given below and omit the "docker" part.

## Datasets

This repository was used against two different datasets: a generated one "Moving Digits", and an industrial one "UTCam".
Hereafter the steps to get each datasets are detailed.

**Generation of mnist data**

The Moving Digit dataset is not available for download and needs to be generated.
The generation requires a python module from [this repository](https://github.com/D3lt4lph4/mpeg4part2_extract_compressed).

Once the module is installed, run the following commands:

```bash
mkdir -p data/01_linear_0-9/01_000/

pip install tqdm opencv-python tensorflow h5py

python scripts/99_mnist_generation.py data/ config/99_datageneration/orientation/01_config_base.json -ns <num_samples> -mp
```

The dataset will be generated in `data/`.

Then set files should be created to tell which folder to use for training/validation/testing:

```bash
touch data/01_linear_0-9/01_000/train_set.txt
touch data/01_linear_0-9/01_000/validation_set.txt
touch data/01_linear_0-9/01_000/train_set.txt

# Add these line in their respective files
01_linear_0-9/01_000/train
01_linear_0-9/01_000/validation
01_linear_0-9/01_000/test
```

**Download the UTCam dataset**

The UTCam dataset can be downloaded at the following [link](https://zenodo.org/record/6826009).
Make sure that you meet the license requirements before using the dataset.

## Training the networks

Once the data is ready, the networks can be trained.
For instance, on the Moving Digit dataset:

```bash
mkdir /tmp/experiment

docker container run --gpus all --rm -v /tmp/experiments:/experiments  -v $(pwd):/app:ro -v $(pwd)/data:/data:ro  $(pwd)/flow_estimation:latest scripts/01_training.py config/01_training_confs/01_moving_digits/01_DM3D_000.json
```

Be sure to modify the "set_file" values in the configuration to point to the correct set files.

> *The training scripts uses the DATASET_PATH and EXPERIMENTS_OUTPUT_DIRECTORY to know where to look for the data and were to output the results*
> *They are by default set to /data and /experiments in the dockerfile.*

## Evaluating the networks

Once the network has been trained, it can be evaluated:

```bash
docker container run  -v $(pwd):/app:ro --gpus all -v /tmp/experiments:/experiments -v $(pwd)/data:/data:ro $(pwd)/flow_estimation:latest scripts/02_testing.py /experiments/0001_DM3D_000 /app/config/02_testing_confs/01_moving_digits/mnist_10_000.json
```

## Display some predictions

**Under construction**

```bash
# Add user to authorized
xhost +local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# Run display
docker container run  -v $(pwd):/app:ro \ 
	-v /experiment/folder:/experiments:ro \
	-v /data/folder:/data:ro -e DISPLAY=$DISPLAY \
	-v $XSOCK:$XSOCK \
	-v $XAUTH:$XAUTH \
	-e XAUTHORITY=$XAUTH \
	--rm \
	d3lt4lph4/mpeg4-flow-estimation-display:v1 \
	/path/to/experiment \
	/path/to/experiment/checkpoints/best_weights.h5 \
	/path/to/test_set.txt

# Remove user from authorized
xhost -local:docker
```

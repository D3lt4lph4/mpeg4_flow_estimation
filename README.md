# mpeg4_flow_estimation

## Setup the environment

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
mkdir data tqdm opencv-python tensorflow h5py

pip install 

python scripts/99_mnist_generation.py data/ config/99_datageneration/orientation/01_config_base.json -ns <num_samples> -mp
```

The dataset will be generated in `data/`.

**Download the UTCam dataset**

The UTCam dataset can be downloaded at the following [link](https://zenodo.org/record/6826009).
Make sure that you meet the license requirements before using the dataset.

## Training the networks

## Evaluating the networks

## Display some predictions

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
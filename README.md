# mpeg4_flow_estimation


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
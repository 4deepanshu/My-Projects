UID=20272
GID=20272
HOME_DIR=/localhome/masc00008
CONTAINER_NAME=masc00008Docker
PROJECT_DIRECTORY_HOST=/raid/masc00008/neural-networks-project-vision
PROJECT_DIRECTORY_CONTAINER=/project
DOCKER_IMAGE=masc00008/nnti-t1
JUPYTER_DOCKER_PORT=61242 # forward the jupyter port to this port on the host machine
JUPYTER_PORT=8888
# --rm makes docker delete the container as soon as it is closed, keep that in mind if we need to make changes to the container
docker run --gpus all --ipc=host --name $CONTAINER_NAME --rm \
-u $UID:$GID \
-v $HOME_DIR:$HOME_DIR \
-v /etc/group:/etc/group:ro \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/shadow:/etc/shadow:ro \
-v $PROJECT_DIRECTORY_HOST:$PROJECT_DIRECTORY_CONTAINER \
-p $JUPYTER_DOCKER_PORT:$JUPYTER_PORT \
-w $PROJECT_DIRECTORY_CONTAINER $DOCKER_IMAGE jupyter notebook

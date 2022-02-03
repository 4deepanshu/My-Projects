# +
UID=20272
GID=20272
HOME_DIR=/localhome/masc00008
CONTAINER_NAME=masc00008Docker
PROJECT_DIRECTORY_HOST=/raid/masc00008/neural-networks-project-vision
PROJECT_DIRECTORY_CONTAINER=/project
DOCKER_IMAGE=masc00008/nnti

docker run --gpus all --ipc=host --name $CONTAINER_NAME --rm \
-u $UID:$GID \
-v $HOME_DIR:$HOME_DIR \
-v /etc/group:/etc/group:ro \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/shadow:/etc/shadow:ro \
-v $PROJECT_DIRECTORY_HOST:$PROJECT_DIRECTORY_CONTAINER \
-w $PROJECT_DIRECTORY_CONTAINER/code $DOCKER_IMAGE python $1

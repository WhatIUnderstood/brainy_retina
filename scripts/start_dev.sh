#!/bin/bash
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
    echo "usage: ./start_dev.sh [--opts '<docker run options>']"
    echo " - webcam: ./start_dev.sh --opts \"--device=/dev/video0:/dev/video0\""
    exit 0
    ;;
    --opts)
    OPTS="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--persistent)
    PERSISTENT=YES
    shift # past argument
    ;;
    -n|--name)
    ENV_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    --rebuild)
    REBUILD=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
# set -- "${POSITIONAL[@]}" # restore positional parameters

if [[ ! $ENV_NAME ]]
then
    ENV_NAME="default"
fi

echo "OPTS       = ${OPTS}"
echo "PERSISTENT = ${PERSISTENT}"
echo "ENV_NAME   = ${ENV_NAME}"

# Start the persistent dev docker or reuse it if already launched
DOCKER_CONTAINER_NAME="retina_dev_$ENV_NAME"
DOCKER_IMG_NAME="retina_dev"
DOCKER_IMG_DOCKERFILE="dockerfiles/dev_dockerfile"
DOCKER_CONTEXT_PATH="dockerfiles"

# Get the bash script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Get the root folder of this project
ROOT_DIR=$SCRIPT_DIR/..
DOCKER_IMG_DOCKERFILE_ABS=$ROOT_DIR/$DOCKER_IMG_DOCKERFILE
DOCKER_CONTEXT_PATH_ABS=$ROOT_DIR/$DOCKER_CONTEXT_PATH

DOCK_OPT_DISPLAY_EXEC=" -e DISPLAY=$DISPLAY  --env='DISPLAY' -w $ROOT_DIR"

DOCK_OPT_DISPLAY_RUN=$DOCK_OPT_DISPLAY_EXEC" -v /tmp/.X11-unix:/tmp/.X11-unix --runtime=nvidia --net=host -v $ROOT_DIR:$ROOT_DIR --security-opt seccomp=unconfined $OPTS"

# If the container should not be persistent, add --rm option
if [[ ! $PERSISTENT ]]
then
    DOCK_OPT_DISPLAY_RUN=$DOCK_OPT_DISPLAY_RUN" --rm"
fi

build_docker(){
    docker build -t $DOCKER_IMG_NAME -f $DOCKER_IMG_DOCKERFILE_ABS $DOCKER_CONTEXT_PATH_ABS
}

if [[ $REBUILD ]]
then
    echo "Building docker $DOCKER_CONTAINER_NAME ..."
    build_docker || (echo "failed to build docker image $DOCKER_IMG_NAME" ; exit 1)
fi

echo "Starting $DOCKER_CONTAINER_NAME docker from $ROOT_DIR"

if [ ! "$(docker ps -q -f name=$DOCKER_CONTAINER_NAME)" ]; then
    # The container already exists, either start it or exec
    if [ "$(docker ps -q -a -f name=$DOCKER_CONTAINER_NAME)" ]; then
        echo "Starting $DOCKER_CONTAINER_NAME docker for the project at $ROOT_DIR"
        echo docker start -i --attach $DOCKER_CONTAINER_NAME
        docker start -i --attach $DOCKER_CONTAINER_NAME
    else
        # if image does not exists, build it from local dev dockerfile
        if [[ "$(docker images -q $DOCKER_CONTAINER_NAME 2> /dev/null)" == "" ]]; then
            echo "Building docker $DOCKER_CONTAINER_NAME ..."
            build_docker || (echo "failed to build docker image $DOCKER_IMG_NAME" ; exit 1)
        fi

        # The container does not exists, create it
        echo "Creating $DOCKER_CONTAINER_NAME docker for the project at $ROOT_DIR"
        docker run --name $DOCKER_CONTAINER_NAME $DOCK_OPT_DISPLAY_RUN  -it $DOCKER_IMG_NAME  /bin/bash
    fi
else
    echo "Starting a new process in $DOCKER_CONTAINER_NAME docker for the project at $ROOT_DIR"
    docker exec -i -t $DOCK_OPT_DISPLAY_EXEC $DOCKER_CONTAINER_NAME "/bin/bash"
fi
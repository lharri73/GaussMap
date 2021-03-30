#!/bin/bash

set -e

dataset_dir=0
build=0
TOKEN=""

showHelp(){
cat << EOF
Usage: ./run.sh --dataset_dir [path_to_nuscenes] <h,b>

-h, --help          Show this usage message
-l, --login         Login to to the nvidia container registry. You only have to do this once.
                    See this page to create an oauth token. Only follow the "Get a New API Key for the NGC Registry"
                    <https://docs.nvidia.com/dgx/ngc-registry-for-dgx-user-guide/>
-b, --build         Force to rebuild the docker container
-d, --dataset_dir   Specify location of the nuscenes dataset root

EOF
}

options=$(getopt -l "help,build,dataset_dir:" -o "hbd:" -a -- "$@")
eval set -- "$options"

while true; do
    case $1 in 
        -h|--help)
            showHelp;
            exit 0
            ;;
        -b|--build)
            build=1
            ;;
        -d|--dataset_dir)
            shift;
            dataset_dir=$1
            ;;
        -l|--login)
            shift;
            TOKEN=$1;
            ;;
        --)
            shift;
            break
            ;; 
    esac
    shift
done

if [[ "$login" -eq 1 ]] ; then
    docker login nvcr.io -u '$oauthtoken' -p $TOKEN
fi

if [[ "$build" -eq 1 || "$(docker images -q gaussmap:latest 2> /dev/null)" == "" ]] ; then
    docker build . -t gaussmap:latest
fi

docker run -it --rm \
    --mount type=bind,src="$(pwd)"/config,target=/opt/gaussMap/config \
    --mount type=bind,src="$(pwd)"/include,target=/opt/gaussMap/include \
    --mount type=bind,src="$(pwd)"/pySrc,target=/opt/gaussMap/pySrc \
    --mount type=bind,src="$(pwd)"/scripts,target=/opt/gaussMap/scripts \
    --mount type=bind,src="$(pwd)"/src,target=/opt/gaussMap/src \
    --mount type=bind,src="$(pwd)"/results,target=/opt/gaussMap/results \
    gaussmap:latest

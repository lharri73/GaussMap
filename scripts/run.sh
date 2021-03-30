#!/bin/bash
set -e

usage(){
    echo "Usage: $0 [version] [split] [dataset_dir] [experiment_name]" 1>&2;
    exit 1;
}

if [ $# -ne 4 ]; then
    usage $@
fi

./pySrc/main.py $1 $2 $3 $4

latest=`ls -t results/$4/*.json | head -1`
curTime=`date +"results_%Y-%m-%d_%H-%M-%S"`

python -m nuscenes.eval.detection.evaluate $latest --output_dir results/$4/$curTime --eval_set $2 --dataroot $3 --version $1 --render_curves 0 --config_path config/eval.json

if [ ! -f "results/$4/map.yml" ]; then
    cp config/map.yml results/$4/map.yml
else
    if [[ ! `diff -q config/map.yml results/$4/map.yml` == "" ]]; then
        echo "CONFIG FILE EXISTS AND ARE DIFFERENT!" 1>&2;
    fi
fi

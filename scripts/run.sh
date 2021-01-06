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

python -m nuscenes.eval.detection.evaluate $latest --output_dir results/$4/$curTime --eval_set $2 --dataroot $3 --version $1

if [ ! -f "results/$4/cfg.yml" ]; then
    cp config/cfg.yml results/$4/cfg.yml
else
    echo "Skipping copy of config file...Already exists"
    echo -e "\tHas it changed?"
fi
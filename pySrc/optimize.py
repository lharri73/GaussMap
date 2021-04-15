#!/usr/bin/env python3
import argparse
from GaussMapWrapper import GaussMapWrapper
import yaml
from easydict import EasyDict as edict
import pickle
import os
import scipy
import numpy as np
from tensorboardX import SummaryWriter


i = 0
def parseArgs():
    parser = argparse.ArgumentParser(description='Gauss Map evaluation for the Nuscenes Dataset')
    parser.add_argument("version", choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"])
    parser.add_argument("split", choices=["train", "val", "test", "mini_train", "mini_val"])
    parser.add_argument("dataset_location")
    parser.add_argument("experiment_name")

    return parser.parse_args()

def main():
    argsP = parseArgs()

    centerTrackRes = {}
    writer = SummaryWriter()
    with open("results/CenterTrack/parsed.pkl", "rb") as f:
        parsed = pickle.load(f)
        split = argsP.split
        if argsP.version == 'v1.0-mini':
            split = argsP.split.replace('_', '-')

        centerTrackRes = parsed[split]
    args = (argsP.experiment_name, argsP, writer, centerTrackRes)

    a = (0,1)
    b = (.2,5)
    c = (.1,5)
    res = scipy.optimize.dual_annealing(func, (a,b,c), args)

    writer.close()
    print(res)

def func(x, expName, args, writer, centerTrackRes):
    global i
    config = {}
    with open('config/map.yml', 'r') as f:
        config = yaml.safe_load(f)

    #if not 0 < x[0] < 1: return 255
    #if not .1 < x[1] < 5: return 255
    #if not .1 < x[2] < 5: return 255
    config['adjustFactor'] = float(x[0])
    config['Radar']['StdDev'] = float(x[1])
    config['Radar']['RadCutoff'] = float(x[2])
    print(x)
    

    with open('config/mapTmp.yml', 'w') as f:
        yaml.dump(config, f)

    number = str(i).zfill(4)
    exp_name = args.experiment_name+number
    try:
        gm = GaussMapWrapper(args.version, args.split, args.dataset_location, centerTrackRes, exp_name,'config/mapTmp.yml')
        gm.run()
        gm.serialize()
    except:
        return 255

    os.system(f"python -m nuscenes.eval.detection.evaluate `ls -t results/{exp_name}/*.json | head -1` --output_dir results/{exp_name}/nusc --eval_set {args.split} --dataroot \
                {args.dataset_location} --version {args.version} --render_curves 0 --plot_examples 0 --config_path config/eval.json")

    score = 0
    metrics = {}
    with open(f"results/{exp_name}/nusc/metrics_summary.json", "r") as f:
        dic = yaml.safe_load(f)
        mate = dic['tp_errors']['trans_err']
        mave = dic['tp_errors']['vel_err']
        for label,dicVal in dic['label_tp_errors'].items():
            metrics.update({f"metric/{label}/aVE": 255 if dicVal['vel_err'] == "NaN" else float(dicVal['vel_err'])})
            metrics.update({f"metric/{label}/aTE": 255 if dicVal['trans_err'] == "NaN" else float(dicVal['trans_err'])})
       
        score = 5 * mate + mave

    metrics.update({"metric/mATE": mate})
    metrics.update({"metric/mAVE": mave})

    metrics.update({"ret/score": score})
    writer.add_hparams({
        "adjustmentFactor": float(x[0]),
        "stdDev": float(x[1]),
        "radiusCutoff": float(x[2])
        }, metrics, global_step=i)
    i+=1
    return score


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from GaussMapWrapper import GaussMapWrapper
import yaml
from easydict import EasyDict as edict
import pickle

def parseArgs():
    parser = argparse.ArgumentParser(description='Gauss Map evaluation for the Nuscenes Dataset')
    parser.add_argument("version", choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"])
    parser.add_argument("split", choices=["train", "val", "test", "mini_train", "mini_val"])
    parser.add_argument("dataset_location")
    parser.add_argument("experiment_name")

    return parser.parse_args()

def main():
    args = parseArgs()

    centerTrackRes = {}
    with open("results/CenterTrack/parsed.pkl", "rb") as f:
        parsed = pickle.load(f)
        split = args.split
        if args.version == 'v1.0-mini':
            split = args.split.replace('_', '-')
        centerTrackRes = parsed[split]

    gm = GaussMapWrapper(args.version, args.split, args.dataset_location, centerTrackRes, args.experiment_name)

    gm.run()
    gm.serialize()

if __name__ == "__main__":
    main()
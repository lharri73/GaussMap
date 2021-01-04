#!/usr/bin/env python3
import argparse
from GaussMapWrapper import GaussMapWrapper
import yaml
from easydict import EasyDict as edict
import pickle

def parseArgs():
    parser = argparse.ArgumentParser(description='Gauss Map evaluation for the Nuscenes Dataset')
    parser.add_argument("version", choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"])
    parser.add_argument("split", choices=["train", "val", "test"])
    parser.add_argument("dataset_location")

    return parser.parse_args()

def main():
    args = parseArgs()

    ## set up the version and split...To keep the cmd line arguments simple

    if args.version == 'v1.0-mini':
        split = 'mini_train' if args.split == 'train' else 'mini_val'
    else:
        split = args.split

    centerTrackRes = {}
    with open("results/CenterTrack/parsed.pkl", "rb") as f:
        parsed = pickle.load(f)
        if args.version == 'v1.0-mini':
            tmpSplit = 'mini-train' if args.split == 'train' else 'mini_mini-valval'
        else:
            tmpSplit = args.split
        centerTrackRes = parsed[tmpSplit]

    gm = GaussMapWrapper(args.version, split, args.dataset_location, centerTrackRes)

    gm.run()

if __name__ == "__main__":
    main()
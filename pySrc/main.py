#!/usr/bin/env python3
import argparse
from GaussMapWrapper import GaussMapWrapper

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

    gm = GaussMapWrapper(args.version, split, args.dataset_location)

    gm.run()

if __name__ == "__main__":
    main()
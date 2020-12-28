#!/usr/bin/env python3
import argparse
from pynuscenes.nuscenes_dataset import NuscenesDataset
from tqdm import tqdm

def parseArgs():
    parser = argparse.ArgumentParser(description='Gauss Map evaluation for the Nuscenes Dataset')
    parser.add_argument("version", choices=["mini", "trainval", "test"])
    parser.add_argument("split", choices=["train", "val", "test"])
    parser.add_argument("dataset_location")

    return parser.parse_args()

def main():
    args = parseArgs()
    

if __name__ == "__main__":
    main()
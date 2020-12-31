#!/usr/bin/env python3
import os
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

import pickle

class_name = {
      'car': 0.0, 'truck': 1.0, 'bus': 2.0, 'trailer': 3.0, 
      'construction_vehicle': 4.0, 'pedestrian': 5.0, 'motorcycle': 6.0, 'bicycle': 7.0,
      'traffic_cone': 8.0, 'barrier': 9.0, 'radar': 10.0}

SPLIT_TO_VER={'val': 'v1.0-trainval', 'train': 'v1.0-trainval', 'test': 'v1.0-test', 'mini-val': 'v1.0-mini', 'mini-train': 'v1.0-mini'}

def main():
    if os.path.exists('results/CenterTrack/parsed.pkl'):
        print("Already parsed")
        exit()
    
    parsedList = {}
    for split in tqdm(SPLIT_TO_VER, position=0):
        parsedList.update({split: parseVersion(SPLIT_TO_VER[split], split)})

    with open('results/CenterTrack/parsed.pkl', 'wb') as f:
        print("writing to pickle")
        pickle.dump(parsedList, f)

def parseVersion(version, split):
    if not os.path.exists("results/CenterTrack/{}.json".format(split)):
        print("could not find 'results/CenterTrack/{}.json'...skipping".format(split))
        return None
    nusc = NuScenes(version=version, dataroot='/DATA/datasets/nuscenes', verbose=True)
    camBoxes, meta = load_prediction("results/CenterTrack/{}.json".format(split), 500, DetectionBox, verbose=True)
    objs = {}
    for sample, dic in tqdm(camBoxes.boxes.items(), position=1):
        sampleList = []
        
        s_record = nusc.get('sample', sample)
        sd_record = nusc.get('sample_data', s_record['data']['CAM_FRONT'])
        pose_rec = nusc.get('ego_pose', sd_record['ego_pose_token'])

        for item in dic:
            item = global_to_vehicle(item, pose_rec, True)
            if item.detection_score < .2: continue
            curList = [item.translation[0], item.translation[1], class_name[item.detection_name]]
            sampleList.append(curList)
        
        objs.update({sample: sampleList})
    return objs

if __name__ == "__main__":
    main()
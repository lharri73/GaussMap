from pynuscenes.nuscenes_dataset import NuscenesDataset
from pynuscenes.utils.nuscenes_utils import vehicle_to_global
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from tqdm import tqdm
from gaussMap import GaussMap
import numpy as np

import time
from consts import class_reverse
import copy
import json

from datetime import datetime
import time
import os

from vis import showImage, showFrame, showFrame3d, saveFrame

class GaussMapWrapper:
    def __init__(self, version, split, dataset_dir, centerTrackRes, save_dir):
        self.nusc = NuscenesDataset(dataset_dir, version, split, 'config/nuscenes.yml')
        self.save_dir = save_dir
        self.now = datetime.now()
        self.start = time.time()
        self.meta = {
            'version': version,
            'split': split,
            'date': self.now.strftime("%m/%d/%Y %H:%M:%S")
        }
        self.centerTrack = centerTrackRes
        self.map = GaussMap('config/map.yml')
        self.results = EvalBoxes()
        self.class_reverse = class_reverse
        self.class_reverse.update({0: 'barrier'})

    def run(self):
        seq = 0
        for frame in tqdm(self.nusc):
            ## get the radar data
            radarFrame = frame['radar']

            ## an Nx18 array of radar points (after the transform)
            radarPoints = radarFrame['pointcloud'].points.T
            cameraPoints = np.array(self.centerTrack[frame['sample_token']])

            if radarPoints.shape[0] == 0 or cameraPoints.shape[0] == 0:
                tqdm.write("No radar or camera points for token: {}".format(frame['sample_token']))
                self.map.reset()
                self.results.add_boxes(frame['sample_token'], [])
                continue

            ## rearange to be similar to the bosch radars
            # mask = [0,1,8,9,15,4,12,13]
            mask = [0,1,9,8,15,4,12,13]
            radarPoints = radarPoints[:,mask]
            # radarPoints[:,2] *=-1
            radarPoints[:,3] *=-1

            ## use the pdh0 to emulate the wExist
            tmp = radarPoints[:,4]
            radarPoints[:,4] = 0 ##np.piecewise(tmp, [tmp==0,tmp==1,tmp==2,tmp==3,tmp==4,tmp==5,tmp==6,tmp==7],
                                 ##                [1-0.0 ,1-0.25,1-0.50,1-0.75,1-0.90,1-0.99,1-0.999,1-1.0])
            ## create the heatmap
            start = time.time()
            radarPoints[:,6] = 4 * radarPoints[:,6] / 30.0
            radarPoints[:,7] = 4 * radarPoints[:,7] / 30.0
            self.map.addRadarData(radarPoints)
            radar = time.time()
            # camPoints = np.empty((1,3))
            self.map.addCameraData(cameraPoints)
            camTime = time.time()
            maxima = self.map.associate()
            # showFrame(frame, maxima, 3,seq)
            seq += 1
            assTime = time.time()
            tqdm.write("radar: {:.5f}, camera: {:.5f}, associate: {:.5f}, total: {:.5f}".format(radar-start, camTime-radar, assTime-camTime, assTime-start))
            # if frame['sample_token'] == "0d0700a2284e477db876c3ee1d864668":
            #     print("results", maxima.shape)
            #     print("radarPoints", radarPoints.shape)
            #     print("camPoints", cameraPoints.shape)
            #     self.saveFrame(frame, maxima, cameraPoints)
            #     # self.showImage()
            #     input()

            # ## Handle the case where there are no points found in this frame
            if maxima.shape[0] == 0:
                tqdm.write("No maxima for token: {}".format(frame['sample_token']))
                self.map.reset()
                self.results.add_boxes(frame['sample_token'], [])
                continue
            
            # ## normalize the scores for this frame
            # scores = maxima[:,3]
            # np.savetxt("maxima.txt", maxima)
            # scores = scores / np.max(scores)

            s_record = self.nusc.get('sample', frame['sample_token'])
            sd_record = self.nusc.get('sample_data', s_record['data']['LIDAR_TOP'])
            pose_rec = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

            boxes = []
            for i in range(min(maxima.shape[0], 499)):
                box = DetectionBox(sample_token=frame['sample_token'],
                                   translation=[maxima[i,0], maxima[i,1], 1],
                                   size=[1,1,1], 
                                   rotation=[1,0,0,0], 
                                   velocity=[maxima[i,2],maxima[i,3]], 
                                   detection_name=self.class_reverse[int(maxima[i,4])],
                                   detection_score=1.0)
                box = vehicle_to_global(box, pose_rec)
                boxes.append(box)

            self.results.add_boxes(frame['sample_token'], boxes)
            self.map.reset()
        
        self.end = time.time()
            # self.showFrame(frame)
            
    def serialize(self):
        resString = self.results.serialize()
        self.meta.update({"timeElapsed": time.strftime("%H:%M:%S", time.gmtime(self.end-self.start)) })
        out = {
            "meta" : self.meta,
            "results": resString
        }
        if not os.path.exists("results/{}".format(self.save_dir)):
            os.makedirs("results/{}".format(self.save_dir))

        with open("results/{}/{}.json".format(self.save_dir, self.now.strftime("%Y-%m-%d_%H-%M-%S")), "w") as f:
            json.dump(out, f, indent=4)

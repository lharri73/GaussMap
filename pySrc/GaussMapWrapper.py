from pynuscenes.nuscenes_dataset import NuscenesDataset
from pynuscenes.utils.nuscenes_utils import vehicle_to_global
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.data_classes import EvalBoxes
from tqdm import tqdm
from gaussMap import GaussMap
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import time
from consts import class_reverse
import copy
import json

from datetime import datetime
import time
import os

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
            mask = [0,1,8,9,15,4]
            radarPoints = radarPoints[:,mask]

            ## use the pdh0 to emulate the wExist
            tmp = radarPoints[:,4]
            radarPoints[:,4] = 0 ##np.piecewise(tmp, [tmp==0,tmp==1,tmp==2,tmp==3,tmp==4,tmp==5,tmp==6,tmp==7],
                                 ##                [1-0.0 ,1-0.25,1-0.50,1-0.75,1-0.90,1-0.99,1-0.999,1-1.0])
            ## create the heatmap
            start = time.time()
            self.map.addRadarData(radarPoints)
            radar = time.time()
            self.map.addCameraData(cameraPoints)
            camTime = time.time()
            maxima = self.map.associate()
            self.showImage()
            assTime = time.time()
            tqdm.write("radar: {:.5f}, camera: {:.5f}, associate: {:.5f}, total: {:.5f}".format(radar-start, camTime-radar, assTime-camTime, assTime-start))
            # if frame['sample_token'] == "0d0700a2284e477db876c3ee1d864668":
            #     print("results", maxima.shape)
            #     print("radarPoints", radarPoints.shape)
            #     print("camPoints", cameraPoints.shape)
            #     self.showFrame(frame, maxima)
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
                                   velocity=[0,0], 
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


    def showImage(self):
        """
        Creates an image of the heatmap and displays it as greyscale
        """
        f, axarr = plt.subplots(1,1)
        array = self.map.asArray()
        scaled = np.uint8(np.interp(array, (0, array.max()), (0,255)))
        axarr.imshow(scaled, cmap="gray")
              
        # axarr.scatter(maxima[:,1], maxima[:,0], c=maxima[:,2], cmap='Paired', marker='o', s=(72./f.dpi)**2)

        plt.show(block=False)
        plt.waitforbuttonpress()

    def showFrame(self, frame, results):
        """
        Draws the radar pointcloud and the lidar pointcloud in open3d
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame['lidar']['pointcloud'].points[:3,:].T)
        pcd.paint_uniform_color(np.array([1,0,1])) ## blue
        results_pcd = o3d.geometry.PointCloud()
        # print(results.shape)
        results_pcd.points = o3d.utility.Vector3dVector(results[:,:3])
        results_pcd.paint_uniform_color(np.array([0,0,1]))

        rpcd = o3d.geometry.PointCloud()
        rpcd.points = o3d.utility.Vector3dVector(frame['radar']['pointcloud'].points[:3,:].T)
        rpcd.paint_uniform_color(np.array([1,0,0])) ## Red

        cpcd = o3d.geometry.PointCloud()
        ctArray = copy.deepcopy(np.array(self.centerTrack[frame['sample_token']]))
        ctArray[:,2] = 0.5
        cpcd.points = o3d.utility.Vector3dVector(ctArray)
        cpcd.paint_uniform_color(np.array([0,1,0])) ## green
        
        vis.add_geometry(pcd)
        vis.add_geometry(rpcd)
        vis.add_geometry(cpcd)
        vis.add_geometry(results_pcd)
        ctr = vis.get_view_control()

        # ctr.set_up(np.array([1,0,0]))
        ctr.set_zoom(.2)
        ctr.translate(-40,10)
        
        vis.run()
        vis.destroy_window()
        
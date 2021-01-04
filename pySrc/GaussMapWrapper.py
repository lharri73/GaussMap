from pynuscenes.nuscenes_dataset import NuscenesDataset
from tqdm import tqdm
from gaussMap import GaussMap
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

class GaussMapWrapper:
    def __init__(self, version, split, dataset_dir, centerTrackRes):
        self.nusc = NuscenesDataset(dataset_dir, version, split, 'config/cfg.yml')
        self.centerTrack = centerTrackRes

    def run(self):
        for frame in tqdm(self.nusc):
            ## get the radar data
            radarFrame = frame['radar']

            ## an Nx18 array of radar points (after the transform)
            radarPoints = radarFrame['pointcloud'].points.T
            cameraPoints = np.array(self.centerTrack[frame['sample_token']])

            # np.savetxt("radar.txt", radarPoints[:,:2])

            #TODO: get camera points

            ## create the heatmap
            self.createMap()

            self.map.addRadarData(radarPoints)
            self.showImage(cameraPoints)         
            input()
            self.showFrame(frame)
            # print("done")
            # input()

    def createMap(self):
        """
        initializes the heatmap with config parameters from config file
        See include/gaussMap.hpp for full construction signature
        """
        self.map = GaussMap('config/map.yml')

    def showImage(self, camPoints):
        """
        Creates an image of the heatmap and displays it as greyscale
        """
        f, axarr = plt.subplots(1,4)
        array = self.map.asArray()
        scaled = np.uint8(np.interp(array, (0, array.max()), (0,255)))
        axarr[0].imshow(scaled, cmap="gray")

        self.map.addCameraData(camPoints)
        array = self.map.asArray()
        scaled = np.uint8(np.interp(array, (0, array.max()), (0,255)))
        axarr[1].imshow(scaled, cmap="gray")
               
        maxima = self.map.findMax()
        # np.savetxt("maxima.txt", maxima)
        axarr[2].imshow(scaled, cmap='gray')
        axarr[2].plot(maxima[:,1], maxima[:,0], ',', color='r')

        classes = self.map.classes()
        # cls_scaled = np.uint8(np.interp(classes, (0, array.max()), (0,255)))
        axarr[3].imshow(classes, cmap='Paired')

        plt.show(block=False)

    def showFrame(self, frame):
        """
        Draws the radar pointcloud and the lidar pointcloud in open3d
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame['lidar']['pointcloud'].points[:3,:].T)

        rpcd = o3d.geometry.PointCloud()
        rpcd.points = o3d.utility.Vector3dVector(frame['radar']['pointcloud'].points[:3,:].T)
        rpcd.paint_uniform_color(np.array([1,0,0])) ## Red

        cpcd = o3d.geometry.PointCloud()
        ctArray = np.array(self.centerTrack[frame['sample_token']])
        ctArray[:,2] = 0
        cpcd.points = o3d.utility.Vector3dVector(ctArray)
        cpcd.paint_uniform_color(np.array([0,1,0])) ## Red
        
        # vis.add_geometry(pcd)
        vis.add_geometry(rpcd)
        vis.add_geometry(cpcd)
        ctr = vis.get_view_control()

        # ctr.set_up(np.array([1,0,0]))
        ctr.set_zoom(.2)
        ctr.translate(-40,10)
        
        vis.run()
        vis.destroy_window()
        
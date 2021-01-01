from pynuscenes.nuscenes_dataset import NuscenesDataset
from tqdm import tqdm
from gaussMap import GaussMap
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

class GaussMapWrapper:
    def __init__(self, version, split, dataset_dir, gmConfig):
        self.nusc = NuscenesDataset(dataset_dir, version, split, 'config/cfg.yml')
        self.gmConfig = gmConfig.GaussMap

    def run(self):
        for frame in tqdm(self.nusc):
            ## get the radar data
            radarFrame = frame['radar']

            ## an Nx18 array of radar points (after the transform)
            radarPoints = radarFrame['pointcloud'].points.T
            # np.savetxt("radar.txt", radarPoints[:,:2])

            #TODO: get camera points

            ## create the heatmap
            self.createMap()

            self.map.addRadarData(radarPoints)

            ## vis functions
            self.showImage()
            # self.showDerivImage()
            print(self.map.findMax())
            input()
            # self.showFrame(frame)
            # print("done")
            # input()

    def createMap(self):
        """
        initializes the heatmap with config parameters from config file
        See include/gaussMap.hpp for full construction signature
        """
        self.map = GaussMap(
            self.gmConfig.MapWidth, 
            self.gmConfig.MapHeight,
            self.gmConfig.MapResolution,
            self.gmConfig.Radar.stdDev,
            self.gmConfig.Radar.mean,
            self.gmConfig.Radar.radiusCutoff)

    def showImage(self):
        """
        Creates an image of the heatmap and displays it as greyscale
        """
        f, axarr = plt.subplots(1,3)
        array = self.map.asArray()
        scaled = np.uint8(np.interp(array, (0, array.max()), (0,255)))
        axarr[0].imshow(scaled, cmap="gray")
        # plt.show(block=False)

        first, second = self.map.derivative()
        scaledDeriv = np.uint8(np.interp(first, (0,first.max()), (0,255)))
        axarr[1].imshow(scaledDeriv, cmap="gray")
        scaledsecond = np.uint8(np.interp(second, (0,second.max()), (0,255)))
        axarr[2].imshow(scaledsecond, cmap="gray")

        plt.show(block=False)

    def showDerivImage(self):
        """
        Creates an image of the heatmap's derivative
        """
        array = self.map.derivative()
        scaled = np.uint8(np.interp(array, (0,array.max()), (0,255)))
        plt.imshow(scaled, cmap="gray")
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
        
        vis.add_geometry(pcd)
        vis.add_geometry(rpcd)
        ctr = vis.get_view_control()

        # ctr.set_up(np.array([1,0,0]))
        ctr.set_zoom(.2)
        ctr.translate(-40,10)
        
        vis.run()
        vis.destroy_window()
        
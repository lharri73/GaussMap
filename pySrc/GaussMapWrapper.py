from pynuscenes.nuscenes_dataset import NuscenesDataset
from tqdm import tqdm
from gaussMap import GaussMap
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# def scale(array, min, max):
#     stdDev = (array - array.min() )

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
            # np.savetxt("radar.txt", radarPoints)
            print(radarPoints.shape)

            #TODO: get camera points

            ## create the heatmap
            self.createMap()
            self.map.addRadarData(radarPoints)
            self.showImage()
            # self.showFrame(frame)
            # input()

    def createMap(self):
        ## GaussMap is initialized with (width, height, vcells, hcells)
        self.map = GaussMap(
            self.gmConfig.MapWidth, 
            self.gmConfig.MapHeight,
            self.gmConfig.MapResolution,
            self.gmConfig.Radar.stdDev,
            self.gmConfig.Radar.mean,
            self.gmConfig.Radar.radiusCutoff)

    def showImage(self):
        array = self.map.asArray()
        # np.savetxt("array.txt", array)
        scaled = np.uint8(np.interp(array, (0, array.max()), (0,255)))
        plt.imshow(scaled, cmap="gray")
        plt.show(block=False)
        input()

    def showFrame(self, frame):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame['lidar']['pointcloud'].points[:3,:].T)

        rpcd = o3d.geometry.PointCloud()
        rpcd.points = o3d.utility.Vector3dVector(frame['radar']['pointcloud'].points[:3,:].T)
        rpcd.paint_uniform_color(np.array([1,0,0]))

        o3d.visualization.draw_geometries([pcd, rpcd])


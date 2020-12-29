from pynuscenes.nuscenes_dataset import NuscenesDataset
from tqdm import tqdm
from gaussMap import GaussMap
import numpy as np

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

            #TODO: get camera points

            ## create the heatmap
            self.createMap()
            self.map.addRadarData(radarPoints)
            # input()
            ## call the destructor for python
            self.map.cleanup()
            # input()

    def createMap(self):
        ## GaussMap is initialized with (width, height, vcells, hcells)
        self.map = GaussMap(
            self.gmConfig.MapWidth, 
            self.gmConfig.MapHeight,
            self.gmConfig.MapResolution,
            self.gmConfig.Radar.stdDev,
            self.gmConfig.Radar.mean)

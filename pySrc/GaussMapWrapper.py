from pynuscenes.nuscenes_dataset import NuscenesDataset
from tqdm import tqdm
from gaussMap import GaussMap
import numpy as np
from PIL import Image

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

            #TODO: get camera points

            ## create the heatmap
            self.createMap()
            self.map.addRadarData(radarPoints)
            # self.showImage()
            # input()

    def createMap(self):
        ## GaussMap is initialized with (width, height, vcells, hcells)
        self.map = GaussMap(
            self.gmConfig.MapWidth, 
            self.gmConfig.MapHeight,
            self.gmConfig.MapResolution,
            self.gmConfig.Radar.stdDev,
            self.gmConfig.Radar.mean)

    def showImage(self):
        array = self.map.asArray()
        scaled = np.uint8(np.interp(array, (array.min(), array.max()), (0,255)))
        img = Image.fromarray(scaled, 'L')
        img.show()
        input()
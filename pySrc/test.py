#!/usr/bin/env python3
import numpy as np
from PIL import Image


array = np.array([[1,1,1,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16]])
scaled = np.interp(array, (array.min(), array.max()), (1,16))
print(scaled)

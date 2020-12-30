#!/usr/bin/env python3
## This is just a file used to test a couple of features along the way
## not actually any unit tests...
import numpy as np
from PIL import Image


array = np.array([[1,1,1,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [13,14,15,16]])
scaled = np.interp(array, (array.min(), array.max()), (1,16))
print(scaled)

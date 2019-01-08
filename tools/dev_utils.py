
from scipy.misc import comb
from scipy import optimize
import numpy as np
import math

import os
import time

def rename():
    path = "/media/administrator/deeplearning/detectron/video_image/choose2label/"
    for filename in os.listdir(path):
        newname = "sh_{}_000002_leftImg8bit.png".format(filename[:-4])
        print (filename + " -> " + newname)
        os.rename(path + filename, path + newname)


rename()
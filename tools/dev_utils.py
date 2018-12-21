
from scipy.misc import comb
from scipy import optimize
import numpy as np
import math

import os
import time

def rename():
    path = "/media/administrator/xhpan/ros_img_label_12_12_16/choose/"
    for filename in os.listdir(path):
        newname = "us_{}_000002_leftImg8bit.png".format(filename[:-4])
        print (filename + " -> " + newname)
        os.rename(path + filename, path + newname)


rename()
#!/usr/bin/env python

import rospy
import time
import json
import zmq
from std_msgs.msg import String


class Const:
    class ConstError(TypeError) : pass
    class ConstCaseError(ConstError):pass

    def __setattr__(self, name, value):
            if name in self.__dict__:
                raise self.ConstError, "Can't change const value!"
            if not name.isupper():
                raise self.ConstCaseError, 'const "%s" is not all letters are capitalized' %name
            self.__dict__[name] = value

# import sys
# sys.modules[__name__] = Const()


import const

const.CAMERA_TYPE = 0#0:sh_pc 1:sh_px2 2:us_px2
const.ROAD_TYPE = 0#0:highway 1:cityroad

const.MPH_TO_MS = 0.44704
# const.STEER_RATIO = 13.0
const.STEER_RATIO = 15.3
const.WHEEL_BASE = 2.70

const.PORT_REPLAY_IP = "172.16.20.11"
const.PORT_GPS_IP = "172.16.60.111"
const.PORT_GPS_IN = 6710
const.PORT_GPS_OUT = 11159

const.PORT_DETECTION = 6701
const.PORT_IMAGE_OUT = 6702
const.PORT_DR_OUT = 6704

const.MASK_SAMPLE_STEP = 10
const.ENABLE_FORK = False
const.INTERVAL_FORK = 1


if const.ROAD_TYPE == 1:
    const.MASK_SAMPLE_STEP = 5
const.CAMERA_SH = True
if const.CAMERA_TYPE == 2:
    const.CAMERA_SH = False
    const.INTERVAL_FORK = 3

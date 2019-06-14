import numpy as np
import const
import cv2

class Config(object):

    BOX_SLOPE_LIMITED = 1
    PARABORA_SLOPE_LIMITED = 600./120
    IMAGE_WID = 960
    IMAGE_HEI = 604
    CURVETURE_MAX = 50.0 / (IMAGE_HEI * IMAGE_HEI)

    source_arr = np.float32([[916, 841], [1091, 841], [1103, 874], [903, 874]])
    # source_arr = np.float32([[907, 783], [1085, 783], [1098, 817], [892, 817]])
    source_arr[:, 1] = source_arr[:, 1] - 504
    CUT_OFFSET_IMG = np.array([252, 451])
    if const.CAMERA_TYPE == 1:
        print ("is_px2 is true")
        source_arr = np.float32([[907, 783], [1085, 783], [1098, 817], [893.5, 817]])
        source_arr[:, 1] = source_arr[:, 1] - 454
        CUT_OFFSET_IMG = np.array([227, 426])
    elif const.CAMERA_TYPE == 2:
        source_arr = np.float32([[852, 860], [1270, 860], [1396, 956], [764, 956]])
        source_arr[:, 1] = source_arr[:, 1] - 540
        CUT_OFFSET_IMG = np.array([270, 478])
    elif const.CAMERA_TYPE == 3:
        pass
        IMAGE_HEI = 540
        CURVETURE_MAX = 50.0 / (IMAGE_HEI * IMAGE_HEI)

        source_arr = np.float32([[750, 644], [1240, 644], [1368, 720], [638, 720]])
        source_arr[:, 1] = source_arr[:, 1] - 472
        CUT_OFFSET_IMG = np.array([236, 374])

    scale_size = (const.CAMERA_TYPE != 2)

    mileage_trigger = 30 if const.ROAD_TYPE == 0 else 20
    undistort_param = np.array([[-0.35262804, 0.15311474, 0.00038879, 0.00048328, - 0.03534825]])
    camera_mtx = np.array([[980.76745978, 0., 969.74796847], [0., 984.13242608, 666.25746185], [0., 0., 1.]])

    scale_rate = 1920. / IMAGE_WID
    source_arr = source_arr / scale_rate
    lane_wid = 200 / scale_rate


    def __init__(self):
        scale_h = 0.025
        scale_w = 0.28#1/3.5
        if const.CAMERA_TYPE == 2:
            scale_h = 0.1 if const.ROAD_TYPE == 0 else 0.8
            scale_w = 1
        elif const.CAMERA_TYPE == 3:
            scale_h = 0.1
            scale_w = 1

        offset_x = self.lane_wid * scale_w / 2
        offset_y = 1 - scale_h


        dest_arr = np.float32([[self.IMAGE_WID / 2 - offset_x, self.IMAGE_HEI * offset_y],
                               [self.IMAGE_WID / 2 + offset_x, self.IMAGE_HEI * offset_y],
                               [self.IMAGE_WID / 2 + offset_x, self.IMAGE_HEI - 1],
                               [self.IMAGE_WID / 2 - offset_x, self.IMAGE_HEI - 1]])

        self.H = cv2.getPerspectiveTransform(self.source_arr, dest_arr)
        self.H_OP = cv2.getPerspectiveTransform(dest_arr, self.source_arr)

        print ("self.H", self.H)
        print ("self.H_OP", self.H_OP)

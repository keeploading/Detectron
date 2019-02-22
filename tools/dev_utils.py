
from scipy.misc import comb
from scipy import optimize
import numpy as np
import math

import os
import time
import json

def rename():
    path = "/media/administrator/deeplearning/detectron/video_image/cityroad/choose/"
    for filename in os.listdir(path):
        newname = "shcity_{}_000002_leftImg8bit.png".format(filename[:-4])
        print (filename + " -> " + newname)
        os.rename(path + filename, path + newname)
rename()
def filter_line():

    file = "/media/administrator/deeplearning/detectron/output/pr/log_20190117"
    f = open(file, 'r')
    lines = f.readlines()
    filter_lines = []
    for line in lines:
        if line.startswith("json_stats"):
            line = line[12:]
            filter_lines.append(line)
    f.close()
    file = file + "_json.txt"
    f = open(file, "w")
    f.writelines(filter_lines)
    f.close()
    print ("finish write file :{}".format(file))



def loss_curve():

    file = "/media/administrator/deeplearning/detectron/output/pr/log_20190117_json.txt"
    f = open(file, 'r')
    lines = f.readlines()
    # iters = []
    # accuracy_cls = []
    # loss_total = []
    # loss_bbox = []
    # loss_cls = []
    # loss_mask = []
    # loss_rpn_bbox_fpn23456 = []
    # loss_rpn_cls_fpn23456 = []
    # for line in lines:
    #     line_json = json.load(line)
    #     accuracy_cls.append(line_json['accuracy_cls'])
    #     iters.append(line_json['iter'])
    #     loss_total.append(line_json['loss'])
    #     loss_bbox.append(line_json['loss_bbox'])
    #     loss_cls.append(line_json['loss_cls'])
    #     loss_mask.append(line_json['loss_mask'])
    #     loss_rpn_bbox_fpn23456.append([line_json['loss_rpn_bbox_fpn2'], line_json['loss_rpn_bbox_fpn3'], line_json['loss_rpn_bbox_fpn4'], line_json['loss_rpn_bbox_fpn5'], line_json['loss_rpn_bbox_fpn6']])
    #     loss_rpn_cls_fpn23456.append([line_json['loss_rpn_cls_fpn2'], line_json['loss_rpn_cls_fpn3'], line_json['loss_rpn_cls_fpn4'], line_json['loss_rpn_cls_fpn5'], line_json['loss_rpn_cls_fpn6']])

    f.close()
    cvs_file = file[0:-4] + "_cvs.txt"
    cvs_lines = []
    f = open(cvs_file, "w")
    cvs_lines.append("accuracy_cls,iter,loss,loss_bbox,loss_cls,loss_mask,loss_rpn_bbox_fpn2,loss_rpn_bbox_fpn3,loss_rpn_bbox_fpn4,loss_rpn_bbox_fpn5,loss_rpn_bbox_fpn6,"
            "loss_rpn_cls_fpn2,loss_rpn_cls_fpn3,loss_rpn_cls_fpn4,loss_rpn_cls_fpn5,loss_rpn_cls_fpn6,lr\n")

    for line in lines:
        line_json = json.loads(line)
        cvs_lines.append("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(line_json['accuracy_cls'],
                                                                         line_json['iter'],
                                                                         line_json['loss'],
                                                                         line_json['loss_bbox'],
                                                                         line_json['loss_cls'],
                                                                         line_json['loss_mask'],
                                                                         line_json['loss_rpn_bbox_fpn2'],
                                                                         line_json['loss_rpn_bbox_fpn3'],
                                                                         line_json['loss_rpn_bbox_fpn4'],
                                                                         line_json['loss_rpn_bbox_fpn5'],
                                                                         line_json['loss_rpn_bbox_fpn6'],
                                                                         line_json['loss_rpn_cls_fpn2'],
                                                                         line_json['loss_rpn_cls_fpn3'],
                                                                         line_json['loss_rpn_cls_fpn4'],
                                                                         line_json['loss_rpn_cls_fpn5'],
                                                                         line_json['loss_rpn_cls_fpn6'],
                                                                         line_json['lr']
                                                                                       ))
        # break
    f.writelines(cvs_lines)
    f.close()
    print ("finish write file :{}".format(cvs_file))
# filter_line()
# loss_curve()
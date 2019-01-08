# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Detection output visualization module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os

import pycocotools.mask as mask_util

from detectron.utils.colormap import colormap
import detectron.utils.env as envu
import detectron.utils.keypoints as keypoint_utils
import detectron.datasets.dummy_datasets as dummy_datasets

# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
envu.set_up_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import optimize
import math
import arp.const as const
import arp.lane_line as lane_line
from arp.lane_line import Line
import time
import arp.math_utils as math_utils

from multiprocessing import Process, Manager

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
scale_size = (const.CAMERA_TYPE != 2)
is_px2 = (const.CAMERA_TYPE == 1)

IMAGE_WID = 960
IMAGE_HEI = 604
source_arr = np.float32([[918,841],[1092,841],[1103,874],[903,874]])
source_arr[:,1] = source_arr[:,1] - 604
if const.CAMERA_TYPE == 1:
    print ("is_px2 is true")
    source_arr = np.float32([[907, 783],[1085,783],[1098,817],[892,817]])
    source_arr[:,1] = source_arr[:,1] - 554
elif const.CAMERA_TYPE == 2:
    source_arr = np.float32([[852, 860], [1270, 860], [1396, 956], [764, 956]])
    source_arr[:,1] = source_arr[:,1] - 670


scale_rate = 1920./IMAGE_WID

source_arr = source_arr / scale_rate
# source_arr = np.float32([[459. , 420.5],[546. , 420.5],[551.5, 437. ],[451.5, 437. ]])
lane_wid = 200 / scale_rate
BOX_SLOPE_LIMITED = 1
PARABORA_SLOPE_LIMITED = 600./120
scale_h = 0.025
scale_w = 0.28#1/3.5
mileage_trigger = 30 if const.ROAD_TYPE == 0 else 20
if const.CAMERA_TYPE == 2:
    scale_h = 0.1 if const.ROAD_TYPE == 0 else 0.5
    scale_w = 1

offset_x = lane_wid * scale_w / 2
offset_y = 1 - scale_h
dest_arr = np.float32([[IMAGE_WID / 2 - offset_x, IMAGE_HEI * offset_y],
                  [IMAGE_WID / 2  + offset_x, IMAGE_HEI * offset_y],
                        [IMAGE_WID / 2 + offset_x, IMAGE_HEI - 1],
                         [IMAGE_WID / 2 - offset_x, IMAGE_HEI - 1]])
H = cv2.getPerspectiveTransform(source_arr, dest_arr)
H_OP = cv2.getPerspectiveTransform(dest_arr, source_arr)
print ("source_arr:" + str(source_arr))
print ("dest_arr:" + str(dest_arr))
print ("lane_wid:" + str(lane_wid))

dist = np.array([[-0.35262804, 0.15311474, 0.00038879, 0.00048328, - 0.03534825]])
mtx = np.array([[980.76745978, 0., 969.74796847], [0., 984.13242608, 666.25746185], [0., 0., 1.]])
CURVETURE_MAX = 50.0/(IMAGE_HEI*IMAGE_HEI)

manager = Manager()
masks_list = manager.dict()
fork_endtime = None
fork_pos = None

def get_detection_line(im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
        show_box=True, dataset=None, show_class=False, frame_id = 0, img_debug = False):
    global fork_endtime
    """Constructs a numpy array with the detections visualized."""

    im = np.array(im)
    mid_im = None
    perspective_img = None
    if img_debug:
        mid_im = np.zeros(im.shape, np.uint8)
        perspective_img = np.zeros((IMAGE_HEI,IMAGE_WID,3), np.uint8)
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)
    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        print ("detection not good!")
        #return None
        return im, mid_im, perspective_img, None, None

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)
        color_list = colormap()
        mask_color_id = 0

    # perspective
    masks_list.clear()
    t = time.time()
    line_class = dummy_datasets.get_line_dataset()

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)


    curve_objs = []
    # del curve_objs[:]

    t = time.time()
    color_index = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        # show box (off by default)

        if show_box and img_debug:
            im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        class_str = get_class_string(classes[i], score, dataset)
        # show class (off by default)
        if show_class and img_debug:
            im = vis_class(im, (bbox[2], bbox[3] - 2), class_str)

        # show mask
        if segms is not None and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            type = ' '.join(class_str.split(' ')[:-1])
            color = dummy_datasets.get_color_dataset(type)
            # if not color is None:
            #     color_mask = color
            if img_debug and (type in line_class):
                mid_im = vis_roi(mid_im, masks[..., i], color_mask)

            if img_debug:
                perspective_color = dummy_datasets.get_perspective_color(color_index)
                im, perspective_img = vis_mask(im, perspective_img, curve_objs, masks[..., i], color_mask, perspective_color, type, score)
                color_index += 1
            elif type in line_class:
                t_find_curve = time.time()
                build_curve_objs(curve_objs, masks[..., i], type, score, img_debug)
                print ("build_curve_objs time:{}".format(time.time() - t_find_curve) )

    print ('loop for build_curve_objs time: {:.3f}s, frame_id:{}'.format(time.time() - t, frame_id))
    parabola_params = optimize_parabola(perspective_img, curve_objs, img_debug, frame_id)
    if parabola_params is None:
        return im, mid_im, perspective_img, None, None
    parabola_params, fork_pos = parabola_params
    return im, mid_im, perspective_img, parabola_params, fork_pos

def optimize_parabola(perspective_img, curve_objs, img_debug, frame_id):
    global fork_pos, fork_endtime
    if len(curve_objs) == 0:
        return
    parabola_params = []
    parabola_box = []
    left_boundary = None
    right_boundary = None
    classes_param = []

    t = time.time()
    new_fork_pos = None
    print ("len(curve_objs):" + str(len(curve_objs)))

    for curve_obj in curve_objs:
        length = len(curve_obj["points"])
        curve_type = curve_obj["classes"]
        print ("curve_type:{}".format(curve_type))
        curve = np.array(np.array(curve_obj["points"]))#[30:length-20, 0:2])
        # curve = curve[0: len(curve): 10]
        curve = curve[curve[:,1].argsort()]
        # curve = curve - [im.shape[1]/2, 0]
        # curve = curve - [0, im.shape[0]/2]
        middle = (curve_obj["end_x_right"] + curve_obj["start_x_left"]) / 2
        max_y = max(curve[:,1])
        min_y = min(curve[:,1])
        middle_y = (max_y + min_y) / 2
        offset_y = max_y - min_y
        offset_x = curve_obj["end_x_right"] - curve_obj["start_x_left"]
        if curve_type == lane_line.FORK_LINE:
            new_fork_pos = [middle - 10, middle_y] if middle < 0 else [middle + 10, middle_y]
            print ("fork_pos:{}".format(new_fork_pos))
            continue
        if length < 3 or offset_y < 50:
            print ("number of points not much!" + str(length))
            continue

        if curve_type != lane_line.BOUNDARY or curve_type == lane_line.FORK_LINE:
            pass
        else:
            if offset_x > lane_wid/2 and float(offset_y) / offset_x < BOX_SLOPE_LIMITED:
                continue
            if curve_obj["mileage"] < 30:
                continue
        parabola_A, parabolaB, parabolaC = optimize.curve_fit(math_utils.parabola2, curve[:, 1], curve[:, 0])[0]
        belive = length + 300. / (curve_obj["mileage"]/length)
        # print ("mileage:" + str(curve_obj["mileage"]/length))
        parabola_param = [parabola_A, parabolaB, parabolaC, curve_obj["score"], belive, middle, min_y + offset_y/2]#, curve_obj["classes"]
        print ("parabola_param:{}".format(parabola_param))
        parabola_params.append(parabola_param)
        classes_param.append(curve_type)
        parabola_box.append([curve_obj["start_x_left"], min_y, curve_obj["end_x_right"], max_y])

        if Line.isBlockLine(curve_type) and parabolaC < 0:
            if (left_boundary is None) or \
                    ((not left_boundary is None) and left_boundary[-2] < parabola_param[-2]):
                adjust_x = (curve_obj["start_x_right"] + curve_obj["end_x_right"]) / 2
                # adjust_x = curve_obj["end_x_right"] - curve_obj["end_x_left"]
                # adjust_x = curve_obj["end_x_right"]
                # parabola_param[2] += (adjust_x - parabola_param[-1]) #boundary left edge
                # parabola_param[2] = adjust_x #boundary left edge
                parabola_param[5] = adjust_x
                left_boundary = parabola_param
        if Line.isBlockLine(curve_type) and parabolaC > 0:
            if (right_boundary is None) or \
                    ((not right_boundary is None) and right_boundary[-2] > parabola_param[-2]):
                adjust_x = (curve_obj["start_x_left"] + curve_obj["end_x_left"]) / 2
                # adjust_x = curve_obj["start_x_right"] - curve_obj["start_x_left"]
                # adjust_x = curve_obj["start_x_left"]
                # parabola_param[2] += (adjust_x - parabola_param[-1])
                #parabola_param[2] = adjust_x
                parabola_param[5] = adjust_x
                right_boundary = parabola_param
    parabola_param_np = np.array(parabola_params)
    classes_param = np.array(classes_param)
    parabola_box = np.array(parabola_box)

    boundarys = [[left_boundary, right_boundary]]

    is_fork = (not new_fork_pos is None)
    if not new_fork_pos is None:
        fork_endtime = time.time() + const.INTERVAL_FORK
        fork_pos = new_fork_pos
        is_fork = True
    elif (not fork_endtime is None) and time.time() < fork_endtime:
        is_fork = True
    else:
        is_fork = False

    is_fork_enable = const.ENABLE_FORK
    if is_fork and not is_fork_enable and len(parabola_param_np) > 0:
        keep_index = None
        if fork_pos[0] < 0:
            keep_index = parabola_param_np[:, -2] >= fork_pos[0]
        else:
            print ("parabola_param_np:{}".format(parabola_param_np))
            keep_index = parabola_param_np[:, -2] < fork_pos[0]
        parabola_param_np = parabola_param_np[keep_index]
        classes_param = classes_param[keep_index]
        parabola_box = parabola_box[keep_index]

    parabola_param_list = []
    classes_param_list = []
    parabola_box_list = []

    if is_fork_enable and is_fork:
        left = []
        right = []
        for index in range(len(parabola_param_np)):
            if classes_param[index] != lane_line.BOUNDARY:
                continue
            if parabola_param_np[index][-2] < fork_pos[0]:
                left.append(parabola_param_np[index])
            else:
                right.append(parabola_param_np[index])
        left_center = fork_pos[0] - lane_wid/2
        right_center = fork_pos[0] + lane_wid/2
        left_left_boundary = None
        left_right_boundary = None

        right_left_boundary = None
        right_right_boundary = None
        for b in left:
            if (b[-2] < left_center) and ((left_left_boundary == None) or (left_left_boundary[-2] < b[-2])):
                left_left_boundary = b.tolist()
            if (b[-2] > left_center) and ((left_right_boundary == None) or (left_right_boundary[-2] > b[-2])):
                left_right_boundary = b.tolist()

        for b in right:
            if (b[-2] < right_center) and ((right_left_boundary == None) or (right_left_boundary[-2] < b[-2])):
                right_left_boundary = b.tolist()
            if (b[-2] > right_center) and ((right_right_boundary is None) or (right_right_boundary[-2] > b[-2])):
                right_right_boundary = b.tolist()
        boundarys = [[left_left_boundary, left_right_boundary],[right_left_boundary, right_right_boundary]]

        left_fork_index = parabola_param_np[:, -2] <= fork_pos[0]
        right_fork_index = parabola_param_np[:, -2] > fork_pos[0]

        parabola_param_list.append(parabola_param_np[left_fork_index])
        parabola_param_list.append(parabola_param_np[right_fork_index])
        classes_param_list.append(classes_param[left_fork_index])
        classes_param_list.append(classes_param[right_fork_index])
        parabola_box_list.append(parabola_box[left_fork_index])
        parabola_box_list.append(parabola_box[right_fork_index])
    else:
        parabola_param_list.append(parabola_param_np)
        classes_param_list.append(classes_param)
        parabola_box_list.append(parabola_box)

    ret_parabola = []
    for parabola_param_np, classes_param, parabola_box, boundary in zip(parabola_param_list, classes_param_list, parabola_box_list, boundarys):
        if not boundary[0] is None:#left
            keep_index = parabola_param_np[:,2] >= boundary[0][2]
            log1 = np.round(parabola_param_np, decimals=1)
            log2 = str(np.round(boundary[0], decimals=1))
            print ("classes_param parabola_param_np:{} ---------------- boundary:{}".format(str(log1), str(log2)))
            print ("{} classes_param:{} ---> keep_index:{}".format(frame_id, classes_param, keep_index))
            parabola_param_np = parabola_param_np[keep_index]
            classes_param = classes_param[keep_index]
            parabola_box = parabola_box[keep_index]
        if not boundary[1] is None:#right
            keep_index = parabola_param_np[:,2] <= boundary[1][2]
            parabola_param_np = parabola_param_np[keep_index]
            classes_param = classes_param[keep_index]
            parabola_box = parabola_box[keep_index]
        # parabola_param_np = parabola_param_np[:,0:3]
        for tmp_box in parabola_box:
            cv2.rectangle(perspective_img, (int(tmp_box[0] + IMAGE_WID/2), int(tmp_box[1] + IMAGE_HEI)), (int(tmp_box[2] + IMAGE_WID/2), int(tmp_box[3] + IMAGE_HEI)), (100, 0, 0), 2)

        ret = get_good_parabola(parabola_param_np, parabola_box)
        if ret is None:
            print ("errer: bad frame detection, didn't find good parabola!")
            continue
        good_parabola, index_param = ret
        curve = np.arange(-IMAGE_HEI, 0, 10)
        x_log = []
        for index, parabola in enumerate(parabola_param_np):
            if index == index_param:
                print ("good_parabola:{}, parabola:{}".format(good_parabola, parabola))
                if img_debug:
                    x_log.append(parabola[2])
                    y = parabola[0] * curve * curve + parabola[1] * curve + parabola[2]
                    color = (100, 0, 20)
                    cv2.polylines(perspective_img, np.int32([np.vstack((y + IMAGE_WID/2, curve + IMAGE_HEI)).T]), False, color, thickness=2)
            else:
                # predict_parabola = parabola[0:3]
                y = get_parabola_y(good_parabola, parabola[-1])
                relative_point = (parabola[-1], y)
                relative_predict = (parabola[-1], get_parabola_y(parabola[0:3], parabola[-1]))
                predict_parabola = get_parabola_by_distance(good_parabola, parabola[5] - y)
                if predict_parabola is None:
                    continue
                parabola_param_np[index][0:3] = predict_parabola
                if parabola_param_np[index][0] == 0:
                    pass
                if img_debug:
                    color = (255, 255, 255)
                    x_log.append(predict_parabola[2])
                    y = predict_parabola[0] * curve * curve + predict_parabola[1] * curve + predict_parabola[2]
                    cv2.polylines(perspective_img, np.int32([np.vstack((y + IMAGE_WID/2, curve + IMAGE_HEI)).T]), False, color, thickness=2)
                    cv2.rectangle(perspective_img, (int(relative_point[1] + IMAGE_WID/2 - 5), int(relative_point[0] + IMAGE_HEI - 5)), (int(relative_point[1] + IMAGE_WID/2 + 5), int(relative_point[0] + IMAGE_HEI + 5)), (255, 0, 0), 2)
                    cv2.rectangle(perspective_img, (int(relative_predict[1] + IMAGE_WID/2 - 5), int(relative_predict[0] + IMAGE_HEI - 5)), (int(relative_predict[1] + IMAGE_WID/2 + 5), int(relative_predict[0] + IMAGE_HEI + 5)), color, 2)
        ret_parabola.append([parabola_param_np, classes_param])
    if len(ret_parabola) == 0:
        return None
    if img_debug and is_fork and not fork_pos is None:
        triangle_size = 10
        triangle_center = [fork_pos[0] + IMAGE_WID/2, fork_pos[1] + IMAGE_HEI]
        triangle_points = np.array([(triangle_center[0], triangle_center[1] - triangle_size), (triangle_center[0] + triangle_size, triangle_center[1] - 2*triangle_size),
                                    (triangle_center[0], triangle_center[1] + 2*triangle_size), (triangle_center[0] - triangle_size, triangle_center[1] - 2*triangle_size),
                                    (triangle_center[0], triangle_center[1] - triangle_size)])
        print ("triangle_points:{}".format(triangle_points))
        cv2.putText(perspective_img, "forked road detected !", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 1)
        cv2.polylines(perspective_img, np.int32([np.vstack((triangle_points[:,0], triangle_points[:,1])).T]), False, (0, 255, 0))
    print ("x_log parabola_param_np:" + str(x_log))
    print ('get parabola_param_np time: {:.3f}s'.format(time.time() - t))
    return ret_parabola, (fork_pos if is_fork_enable else None)

def vis_mask(img, perspective_img, curve_objs,  mask, col, perspective_color, classs_type, score, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    color = dummy_datasets.get_color_dataset(classs_type)
    # if not color is None:
    #     col = color
    img = img.astype(np.float32)

    line_class = dummy_datasets.get_line_dataset()
    if classs_type in line_class:
        perspective_img = perspective_img.astype(np.float32)
        mask, top_idx = build_curve_objs(curve_objs, mask, classs_type, score, True)
        # perspective_img[top_idx[0], top_idx[1], :] *= 1.0 - alpha
        # perspective_img[top_idx[0], top_idx[1], :] += alpha * col
        perspective_img[top_idx[1], top_idx[0], :] = perspective_color
        # perspective_img[top_idx[0], top_idx[1], :] = [255, 255, 255]


    idx = np.nonzero(mask)
    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8), perspective_img.astype(np.uint8)

def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')

def get_class(class_index, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text

def get_closest_points(points, y):
    index_close = -1
    max_distance = 1000
    for index, point in enumerate(points):
        dis = abs(point[1] - y)
        if max_distance > dis:
            max_distance = dis
            index_close = index
    if index_close == 0:
        return [points[0], points[1]]
    elif index_close == len(points) - 1:
        return [points[-1], points[-2]]
    else:
        dis1 = abs(points[index_close + 1][1] - y)
        dis2 = abs(points[index_close - 1][1] - y)
        if dis1 > dis2:
            return [points[index_close], points[index_close - 1]]
        else:
            return [points[index_close], points[index_close + 1]]

def add2curve(curve_objs, point, type, score):
    matched = False
    for i in range(len(curve_objs)):
        obj = curve_objs[i]
        points = obj["points"]
        distanceX = None
        p1 = points[0]
        p2 = points[-1]
        distanceX = min(abs(p1[0] - point[0]), abs(p2[0] - point[0]))
        distanceY1 = p1[1] - point[1]
        distanceY2 = p2[1] - point[1]
        if p1[0] != p2[0] and abs(p1[0] - p2[0] > IMAGE_HEI / 6):
            # p1, p2 = get_closest_points(points, point[1])
            param = math_utils.line_param(p1[1], p1[0], p2[1], p2[0])#right-down to down-right
            distanceX = abs(point[0]- (param[0] * point[1] + param[1]))
            distanceY1 = p1[1] - point[1]
            distanceY2 = p2[1] - point[1]

        if distanceY1 < 0 and distanceY2 > 0:
            if distanceX < 10:
                return
            else:
                continue
        if abs(distanceY1) > IMAGE_HEI / 3 and abs(distanceY2) > IMAGE_HEI / 3:
            # print ("distanceY1:{}, distanceY2:{}, reference:{}".format(distanceY1, distanceY2, IMAGE_HEI / 3))
            continue

        if distanceX < 20 and (obj["classes"] == type or dummy_datasets.isLaneLine(obj["classes"], type)) :
            points.append(point)
            obj["mileage"] += (point[3] - point[2])
            if obj["start_x_left"] > point[2]:
                obj["start_x_left"] = point[2]
            if obj["start_x_right"] < point[2]:
                obj["start_x_right"] = point[2]
            if obj["end_x_right"] < point[3]:
                obj["end_x_right"] = point[3]
            if obj["end_x_left"] > point[3]:
                obj["end_x_left"] = point[3]
            #process share obj
            curve_objs[i] = obj
            matched = True


    if matched:
        if const.CLUSTER_DEBUG:
            debug_im = np.zeros((IMAGE_HEI, IMAGE_WID, 3), np.uint8)
            for _, pts in enumerate(curve_objs):
                debug_points = pts["points"]
                for p in debug_points:
                    cv2.circle(debug_im, (int(p[0] + IMAGE_WID / 2), int(p[1] + IMAGE_HEI)), 5, tuple(const.DEBUG_CORLOR[_]),
                               thickness=5)
            cv2.imwrite("/media/administrator/deeplearning/detectron/debug/{}.png".format(int(round(time.time() * 1000))), debug_im)
        return
    curve = {"points":[point], "start_x_left":point[2], "end_x_right":point[3], "start_x_right":point[2],
             "end_x_left":point[3], "classes": type, "score":score, "mileage": point[3] - point[2]}
    curve_objs.append(curve)

    if const.CLUSTER_DEBUG:
        debug_im = np.zeros((IMAGE_HEI, IMAGE_WID, 3), np.uint8)
        for _, pts in enumerate(curve_objs):
            debug_points = pts["points"]
            for p in debug_points:
                cv2.circle(debug_im, (int(p[0] + IMAGE_WID / 2), int(p[1] + IMAGE_HEI)), 5, tuple(const.DEBUG_CORLOR[_]), thickness=5)
        cv2.imwrite("/media/administrator/deeplearning/detectron/debug/{}.png".format(int(round(time.time() * 1000))), debug_im)
def build_curve_objs(curve_objs, mask, classs_type, score, img_debug):
    line_class = dummy_datasets.get_line_dataset()
    top_idx = None
    if classs_type in line_class:
        # mask = cv2.undistort(mask, mtx, dist, None)
        mask_hei = int(len(mask) / const.MASK_SAMPLE_STEP)
        dalta_list = np.ones(const.MASK_SAMPLE_STEP * mask_hei, dtype=int)
        for i in range(const.MASK_SAMPLE_STEP):
            dalta_list[i*mask_hei: i*mask_hei + mask_hei] = i
        mask_index = 0
        for i in range(0, len(mask) - const.MASK_SAMPLE_STEP):
            if mask_index >= IMAGE_HEI:
                break
            mask[mask_index: mask_index + dalta_list[i]] = 0
            mask_index = mask_index + dalta_list[i] + 1
        # for i in range(0, IMAGE_WID, 5):
        #     mask[:,i: i + 4] = 0

        top_idx = np.nonzero(mask)
        if len(top_idx[0]) < 10:
            return mask, np.array([top_idx[1], top_idx[0]])
        points = np.transpose(np.array([top_idx[1], top_idx[0]]))
        top_idx = cv2.perspectiveTransform(np.array([points], dtype=np.float32), np.array(H))

        top_idx = top_idx[0].astype(np.int32)

        # top_idx = top_idx[::10]
        top_idx = top_idx[(top_idx[:,0] > 0) & (top_idx[:,0] < IMAGE_WID) & (top_idx[:,1] > 0) & (top_idx[:,1] < IMAGE_HEI)]
        # cv2.imshow('log1', mask * 255)
        # cv2.waitKey(10)
        # print('warpPerspective time: {:.3f}s'.format(time.time() - t))
        t = time.time()
        if len(top_idx) > 10:
            y_start = top_idx[0][1]
            x_start = top_idx[0][0]
            x_end = x_start
            for single_point in top_idx:
                if single_point[1] != y_start:
                    add2curve(curve_objs, [(x_end + x_start) / 2 - IMAGE_WID /2, y_start - IMAGE_HEI, x_start - IMAGE_WID /2, x_end - IMAGE_WID /2], classs_type, score)
                    y_start = single_point[1]
                    x_start = single_point[0]
                    x_end = x_start
                else:
                    if single_point[0] - x_end > lane_wid / 10:
                        add2curve(curve_objs, [(x_end + x_start)/2 - IMAGE_WID /2, y_start - IMAGE_HEI, x_start - IMAGE_WID /2, x_end - IMAGE_WID /2], classs_type, score)
                        y_start = single_point[1]
                        x_start = single_point[0]
                        x_end = x_start
                    else:
                        x_end = single_point[0]
            add2curve(curve_objs, [(x_end + x_start) / 2 - IMAGE_WID / 2, y_start - IMAGE_HEI, x_start - IMAGE_WID / 2, x_end - IMAGE_WID / 2], classs_type, score)

                        # print('loop time: {:.3f}s'.format(time.time() - t))
    return mask, [top_idx[:, 0], top_idx[:, 1]]

def vis_roi(img,  mask, col):
    """Visualizes the class."""
    img = img.astype(np.float32)
    idx = np.nonzero(mask)#(1208,1920
    img[idx[0], idx[1], :] *= 1.0 - 0.4
    img[idx[0], idx[1], :] += 0.4 * col
    return img.astype(np.uint8)

def vis_perspective(img):
    """Visualizes the class."""
    img = img.astype(np.float32)
    top_im = cv2.warpPerspective(img, H, (IMAGE_WID, IMAGE_HEI))

    return top_im.astype(np.uint8)

def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0 - int(txt_w), y0 - int(1.3 * txt_h)
    back_br = x0 - int(txt_w) + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    txt_tl = x0 - int(txt_w), y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, thick=4):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img

def undistort_mask(mask, index):
    t = time.time()
    mask = cv2.undistort(mask, mtx, dist, None)
    masks_list[str(index)] = mask
    print ("undistort_mask time:{}".format(time.time() - t) )

def undistort_multiprocess(mask, i):
    process = Process(target=undistort_mask, args=(mask, i))
    process.daemon = True
    return process

def get_good_index(coefficient, suggest_index):
    filter_coeffi = coefficient[suggest_index]
    avg_believe = filter_coeffi[:, 4].tolist()
    good_index = avg_believe.index(np.max(avg_believe))
    total_belive = np.sum(filter_coeffi[:, 4])
    filter_coeffi[:, 4] = filter_coeffi[:, 4] / total_belive
    A = np.sum(filter_coeffi[:, 0] * filter_coeffi[:, 4])
    B = np.sum(filter_coeffi[:, 1] * filter_coeffi[:, 4])
    good_index = suggest_index[good_index]
    coefficient[good_index][0] = A
    coefficient[good_index][1] = B
    return good_index

def get_good_parabola(coefficient, parabola_box):
    good_index = 0
    min_gradient = 1
    perfect_avg_indexs = []
    good_avg_indexs = []
    good_min_indexs = []
    for box_index, box in enumerate(parabola_box):
        hei = box[3] - box[1]
        wid = (box[2] - box[0])
        center = (box[2] + box[0]) / 2
        if (hei > 0.8*IMAGE_HEI and wid < 25) or (hei > 0.5*IMAGE_HEI and wid < 15):
            perfect_avg_indexs.append(box_index)
        elif hei > 0.6*IMAGE_HEI and const.CAMERA_TYPE == 2:#600/60
            good_avg_indexs.append(box_index)
        elif hei > IMAGE_HEI / 2 and hei / (box[2] - box[0]) > 10:#600/60
            good_min_indexs.append(box_index)
    if len(perfect_avg_indexs) > 0:
        good_index = get_good_index(coefficient, perfect_avg_indexs)
    elif len(good_avg_indexs) > 0:
        good_index = get_good_index(coefficient, good_avg_indexs)
    elif len(good_min_indexs) > 0:
        filter_coeffi = coefficient[good_min_indexs]
        avg_wid = filter_coeffi[:,4].tolist()
        good_index = avg_wid.index(np.max(avg_wid))
        good_index = good_min_indexs[good_index]
    else:
        good_box = range(len(coefficient))
        for index in good_box:
            param = coefficient[index]
            gradient1 = get_gradient(param, -IMAGE_HEI)
            gradient2 = get_gradient(param, 0)

            hei = box[3] - box[1]
            wid = box[2] - box[0]
            if abs(gradient2) > (1./PARABORA_SLOPE_LIMITED) and hei < IMAGE_HEI / 3:
                continue
            gradient_dalta = abs(gradient1 - gradient2)



            if gradient_dalta < min_gradient:
                min_gradient = gradient_dalta
                good_index = index
        if min_gradient == 1:
            return None

    box = parabola_box[good_index]
    ratio = (box[3] - box[1])/ IMAGE_HEI
    good_parabola = coefficient[good_index]

    if const.CAMERA_SH:
        y = (3. / 8) * ratio ** 2 + (5. / 8) * ratio  # (0,0)(1,1)
        good_parabola[0:2] = good_parabola[0:2] * y
    return good_parabola, good_index

def get_parabola_y(coefficient, x):
    return coefficient[0] * x * x + coefficient[1] * x + coefficient[2]

def get_gradient(coefficient, x):
    return 2*coefficient[0] * x + coefficient[1]

def get_parabola_by_distance(coefficient, distance):
    A = coefficient[0]
    B = coefficient[1]
    C = coefficient[2]

    if A == 0:
        theta = math.atan2(B, 1)
        return [A, B, C + distance / math.sin(theta)]
    else:
        point1 = [-B/(2*A), (4*A*C - B*B)/(4*A) + distance]

        x_array = [-B/(2*A) + 100, -B/(2*A) + 200]
        source_p1 = [x_array[0], A * x_array[0] * x_array[0] + B * x_array[0] + C]
        source_p2 = [x_array[1], A * x_array[1] * x_array[1] + B * x_array[1] + C]

        theta = math.atan2(2*A*source_p1[0] + B, 1)
        point2 = [source_p1[0] - math.sin(theta) * distance, source_p1[1] + math.cos(theta) * distance]

        theta = math.atan2(2*A*source_p2[0] + B, 1)
        point3 = [source_p2[0] - math.sin(theta) * distance, source_p2[1] + math.cos(theta) * distance]
        #y=ax+b
        if (point3[0] * (point2[1] - point1[1]) + point2[0] * (point1[1] - point3[1]) + point1[0] * (point3[1] - point2[1])) == 0:
            return [0, 0, C + distance]
        return get_parabols_by_points([point1, point2, point3])


def get_parabols_by_points(points):
    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[1][0]
    y2 = points[1][1]
    x3 = points[2][0]
    y3 = points[2][1]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    if C > 960 or C < -960:
        print ("C > 960 or C < -960, please check:" + str(points))
    return [A, B, C]

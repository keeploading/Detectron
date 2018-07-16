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
import logging
import sys
import time
import arp.math_utils as math_utils

from multiprocessing import Process, Manager

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
scale_size = True
is_px2 = False

IMAGE_WID = 1920
IMAGE_HEI = 1208
source_arr = np.float32([[918,841],[1092,841],[1103,874],[903,874]])
source_arr[:,1] = source_arr[:,1] - 604
if is_px2:
    print ("is_px2 is true")
    source_arr = np.float32([[907, 783],[1085,783],[1098,817],[892,817]])
    source_arr[:,1] = source_arr[:,1] - 554
if scale_size:
    IMAGE_WID = 960
    IMAGE_HEI = 604
scale_rate = 1920./IMAGE_WID

source_arr = source_arr / scale_rate
# source_arr = np.float32([[459. , 420.5],[546. , 420.5],[551.5, 437. ],[451.5, 437. ]])
lane_wid = 200 / scale_rate
BOX_SLOPE_LIMITED = IMAGE_HEI/(2. * lane_wid)#2*LANE_WID
PARABORA_SLOPE_LIMITED = 600./40
scale_h = 0.025
scale_w = 0.28
offset_x = lane_wid * scale_w / 2
offset_y = 1 - scale_h
dest_arr = np.float32([[IMAGE_WID / 2 - offset_x, IMAGE_HEI * offset_y],
                       [IMAGE_WID / 2  + offset_x, IMAGE_HEI * offset_y],
                        [IMAGE_WID / 2 + offset_x, IMAGE_HEI - 1],
                         [IMAGE_WID / 2 - offset_x, IMAGE_HEI - 1]])
H = cv2.getPerspectiveTransform(source_arr, dest_arr)
H_OP = cv2.getPerspectiveTransform(dest_arr, source_arr)
print ("source_arr:" + str(source_arr))
print ("lane_wid:" + str(lane_wid))

dist = np.array([[-0.35262804, 0.15311474, 0.00038879, 0.00048328, - 0.03534825]])
mtx = np.array([[980.76745978, 0., 969.74796847], [0., 984.13242608, 666.25746185], [0., 0., 1.]])
CURVETURE_MAX = 50.0/(IMAGE_HEI*IMAGE_HEI)

manager = Manager()
masks_list = manager.dict()


def get_detection_line(im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
        show_box=True, dataset=None, show_class=False, frame_id = 0, img_debug = False):

    """Constructs a numpy array with the detections visualized."""

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)
    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        print ("detection not good!")
        return None

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


    im = np.array(im)
    curve_objs = []
    # del curve_objs[:]
    mid_im = None
    perspective_img = None
    if img_debug:
        mid_im = np.zeros(im.shape, np.uint8)
        perspective_img = np.zeros((IMAGE_HEI,IMAGE_WID,3), np.uint8)

    t = time.time()
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
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)

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
                im, perspective_img = vis_mask(im, perspective_img, curve_objs, masks[..., i], color_mask, type, score)
            elif type in line_class:
                t_find_curve = time.time()
                build_curve_objs(curve_objs, masks[..., i], type, score, img_debug)
                print ("build_curve_objs time:{}".format(time.time() - t_find_curve) )

    print ('loop for build_curve_objs time: {:.3f}s'.format(time.time() - t))

    parabola_params = optimize_parabola(perspective_img, curve_objs, img_debug)
    return im, mid_im, perspective_img, parabola_params

def optimize_parabola(perspective_img, curve_objs, img_debug):
    if len(curve_objs) == 0:
        return
    parabola_params = []
    parabola_box = []
    left_boundary = None
    right_boundary = None
    classes_param = []

    t = time.time()
    print ("len(curve_objs):" + str(len(curve_objs)))
    for curve_obj in curve_objs:
        length = len(curve_obj["points"])
        if length < 5:
            print ("number of points not much!" + str(length))
            continue
        curve = np.array(np.array(curve_obj["points"]))#[30:length-20, 0:2])
        # curve = curve[0: len(curve): 10]
        curve = curve[curve[:,1].argsort()]
        # curve = curve - [im.shape[1]/2, 0]
        # curve = curve - [0, im.shape[0]/2]
        curve_type = curve_obj["classes"]
        middle = (curve_obj["end_x_right"] + curve_obj["start_x_left"]) / 2
        max_y = max(curve[:,1])
        min_y = min(curve[:,1])
        offset_y = max_y - min_y
        offset_x = curve_obj["end_x_right"] - curve_obj["start_x_left"]
        if offset_x > lane_wid/2 and float(offset_y) / offset_x < BOX_SLOPE_LIMITED:
            print ("offset_y:" + str(offset_y) + " offset_x:" + str(offset_x))
            print ("min_y:" + str(min_y) + " max_y:" + str(max_y))
            print ("min_x:" + str(curve_obj["start_x_left"]) + " max_x:" + str(curve_obj["end_x_right"]))
            continue
        parabola_A, parabolaB, parabolaC = optimize.curve_fit(math_utils.parabola2, curve[:, 1], curve[:, 0])[0]
        parabola_param = [parabola_A, parabolaB, parabolaC, curve_obj["score"], curve_obj["mileage"]/length, middle]#, curve_obj["classes"]
        parabola_params.append(parabola_param)
        parabola_box.append([curve_obj["start_x_left"], min_y, curve_obj["end_x_right"], max_y])
        classes_param.append(curve_type)
        if curve_type == "boundary" and curve_obj["start_x_left"] + curve_obj["end_x_right"] < 0:
            if (left_boundary is None) or \
                    ((not left_boundary is None) and left_boundary[-1] < parabola_param[-1]):
                adjust_x = (curve_obj["end_x_right"] + curve_obj["end_x_left"]) / 2
                parabola_param[2] += (adjust_x - parabola_param[-1]) #boundary left edge
                parabola_param[-1] = adjust_x
                left_boundary = parabola_param
        if curve_type == "boundary" and curve_obj["start_x_left"] + curve_obj["end_x_right"] > 0:
            if (right_boundary is None) or \
                    ((not right_boundary is None) and right_boundary[-1] > parabola_param[-1]):
                adjust_x = (curve_obj["start_x_right"] + curve_obj["start_x_left"]) / 2
                parabola_param[2] += (adjust_x - parabola_param[-1])
                parabola_param[-1] = adjust_x
                right_boundary = parabola_param
    parabola_param_np = np.array(parabola_params)
    classes_param = np.array(classes_param)
    parabola_box = np.array(parabola_box)
    if not left_boundary is None:
        keep_index = parabola_param_np[:,-1] >= left_boundary[-1]
        parabola_param_np = parabola_param_np[keep_index]
        classes_param = classes_param[keep_index]
        parabola_box = parabola_box[keep_index]
    if not right_boundary is None:
        keep_index = parabola_param_np[:,-1] <= right_boundary[-1]
        parabola_param_np = parabola_param_np[keep_index]
        classes_param = classes_param[keep_index]
        parabola_box = parabola_box[keep_index]
    # parabola_param_np = parabola_param_np[:,0:3]

    ret = get_good_parabola(parabola_param_np, parabola_box)
    if ret is None:
        print ("errer: bad frame detection, didn't find good parabola!")
        return
    good_parabola, index_param = ret
    curve = np.arange(-IMAGE_HEI, 0, 10)
    for index, parabola in enumerate(parabola_param_np):
        if index == index_param:
            if img_debug:
                y = parabola[0] * curve * curve + parabola[1] * curve + parabola[2]
                color = (100, 0, 20)
                cv2.polylines(perspective_img, np.int32([np.vstack((y + IMAGE_WID/2, curve + IMAGE_HEI)).T]), False, color, thickness=10)
        else:
            # predict_parabola = parabola[0:3]
            predict_parabola = get_parabola_by_distance(good_parabola, parabola[-1] - good_parabola[-1])
            if predict_parabola is None:
                continue
            parabola_param_np[index][0:3] = predict_parabola
            if img_debug:
                color = (255, 255, 255)
                y = predict_parabola[0] * curve * curve + predict_parabola[1] * curve + predict_parabola[2]
                cv2.polylines(perspective_img, np.int32([np.vstack((y + IMAGE_WID/2, curve + IMAGE_HEI)).T]), False, color, thickness=10)

    if not perspective_img is None:
        perspective_img = perspective_img.astype(np.float32)
    print ('get parabola_param_np time: {:.3f}s'.format(time.time() - t))
    return [parabola_param_np, classes_param]

def vis_mask(img, perspective_img, curve_objs,  mask, col, classs_type, score, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    color = dummy_datasets.get_color_dataset(classs_type)
    # if not color is None:
    #     col = color
    img = img.astype(np.float32)

    line_class = dummy_datasets.get_line_dataset()
    if classs_type in line_class:
        perspective_img = perspective_img.astype(np.float32)
        mask, top_idx = build_curve_objs(curve_objs, mask, classs_type, score, True)
        perspective_img[top_idx[0], top_idx[1], :] *= 1.0 - alpha
        perspective_img[top_idx[0], top_idx[1], :] += alpha * col


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

def add2curve(curve_objs, point, type, score):
    matched = False
    for i in range(len(curve_objs)):
        obj = curve_objs[i]
        distance = None
        if abs(point[1] - obj["points"][0][1]) > abs(point[1] - obj["points"][-1][1]):
            distance = abs(obj["points"][-1][0] - point[0])
        else:
            distance = abs(obj["points"][0][0] - point[0])
        if obj["classes"] == type and distance < 50:
            obj["points"].append(point)
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
        return
    curve = {"points":[point], "start_x_left":point[2], "end_x_right":point[3], "start_x_right":point[2],
             "end_x_left":point[3], "classes": type, "score":score, "mileage": point[3] - point[2]}
    curve_objs.append(curve)

def build_curve_objs(curve_objs, mask, classs_type, score, img_debug):
    line_class = dummy_datasets.get_line_dataset()
    top_idx = None
    if classs_type in line_class:
        # mask = cv2.undistort(mask, mtx, dist, None)
        top = cv2.warpPerspective(mask, H, (IMAGE_WID,IMAGE_HEI))
        t_slice = time.time()
        if not img_debug:
            for i in range(0, IMAGE_HEI, 10):
                top[i: i + 9] = 0
        print('loop t_slice time: {:.3f}s'.format(time.time() - t_slice))
        top_idx = np.nonzero(top)
        if len(top_idx[0]) > 100/scale_rate:
            t = time.time()
            # points = np.array(zip(top_idx[0], top_idx[1])) # too expansive
            points = np.transpose(top_idx)
            y_start = points[0][0]
            x_start = points[0][1]
            x_end = x_start
            for single_point in points:
                if single_point[0] != y_start:
                    add2curve(curve_objs, [(x_end + x_start) / 2 - IMAGE_WID /2, y_start - IMAGE_HEI, x_start - IMAGE_WID /2, x_end - IMAGE_WID /2], classs_type, score)
                    y_start = single_point[0]
                    x_start = single_point[1]
                    x_end = x_start
                else:
                    if single_point[1] - x_end > lane_wid / 4:
                        add2curve(curve_objs, [(x_end + x_start)/2 - IMAGE_WID /2, y_start - IMAGE_HEI, x_start - IMAGE_WID /2, x_end - IMAGE_WID /2], classs_type, score)
                        y_start = single_point[0]
                        x_start = single_point[1]
                        x_end = x_start
                    else:
                        x_end = single_point[1]
            print('loop add2curve time: {:.3f}s'.format(time.time() - t))
    return mask, top_idx

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
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
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

def get_good_parabola(coefficient, parabola_box):
    good_parabola = None
    good_index = 0
    min_gradient = 1
    good_box = []
    for box_index, box in enumerate(parabola_box):
        hei = box[3] - box[1]
        if hei > IMAGE_HEI / 2 and hei / (box[2] - box[0]) > 10:#600/60
            good_box.append(box_index)
    if len(good_box) > 0:
        filter_coeffi = coefficient[good_box]
        avg_wid = filter_coeffi[:,4].tolist()
        good_index = avg_wid.index(np.min(avg_wid))
        good_index = good_box[good_index]
    else:
        good_box = range(len(coefficient))
        for index in good_box:
            param = coefficient[index]
            # point1 = get_parabola_y(param, 0)
            # point2 = get_parabola_y(param, -IMAGE_HEI)
            # point3 = get_parabola_y(param, -param[1]/(2*param[0]))
            # point3 = get_parabola_y(param, -IMAGE_HEI/2)
            # points = [point1, point2, point3]
            # dis = max(points) - min(points)
            gradient1 = get_gradient(param, -IMAGE_HEI)
            gradient2 = get_gradient(param, 0)
            if abs(gradient2) > (1./PARABORA_SLOPE_LIMITED):
                continue
            gradient_dalta = abs(gradient1 - gradient2)



            if gradient_dalta < min_gradient:
                min_gradient = gradient_dalta
                good_index = index
        if min_gradient == 1:
            return None
    # if len(good_box) == 0:
    #     good_box = range(len(coefficient))
    # for index in good_box:
    #     param = coefficient[index]
    #     # point1 = get_parabola_y(param, 0)
    #     # point2 = get_parabola_y(param, -IMAGE_HEI)
    #     # point3 = get_parabola_y(param, -param[1]/(2*param[0]))
    #     # point3 = get_parabola_y(param, -IMAGE_HEI/2)
    #     # points = [point1, point2, point3]
    #     # dis = max(points) - min(points)
    #     gradient1 = get_gradient(param, -IMAGE_HEI)
    #     gradient2 = get_gradient(param, 0)
    #     if abs(gradient2) > (1. / PARABORA_SLOPE_LIMITED):
    #         continue
    #     gradient_dalta = abs(gradient1 - gradient2)
    #
    #     if gradient_dalta < min_gradient:
    #         min_gradient = gradient_dalta
    #         good_index = index
    # if min_gradient == 1:
    #     return None


    box = parabola_box[good_index]
    sin_x = (math.pi/2)*((box[3] - box[1])/ IMAGE_HEI) - math.pi/2
    sin_y = math.sin(sin_x) + 1 #[0,1]
    good_parabola = coefficient[good_index]
    good_parabola[0:2] = good_parabola[0:2] * sin_y
    return good_parabola, good_index

def get_parabola_y(coefficient, x):
    return coefficient[0] * x * x + coefficient[1] * x + coefficient[2]

def get_gradient(coefficient, x):
    return 2*coefficient[0] * x + coefficient[1]

def get_parabola_by_distance(coefficient, distance):
    A = coefficient[0]
    B = coefficient[1]
    C = coefficient[2]

    point1 = [-B/(2*A), (4*A*C - B*B)/(4*A) + distance]

    x_array = [-B/(2*A) + 100, -B/(2*A) + 200]
    source_p1 = [x_array[0], A * x_array[0] * x_array[0] + B * x_array[0] + C]
    source_p2 = [x_array[1], A * x_array[1] * x_array[1] + B * x_array[1] + C]

    theta = math.atan2(2*A*source_p1[0] + B, 1)
    point2 = [source_p1[0] - math.sin(theta) * distance, source_p1[1] + math.cos(theta) * distance]

    theta = math.atan2(2*A*source_p2[0] + B, 1)
    point3 = [source_p2[0] - math.sin(theta) * distance, source_p2[1] + math.cos(theta) * distance]
    return get_parabols_by_points([point1, point2, point3])


def get_parabols_by_points(pints):
    x1 = pints[0][0]
    y1 = pints[0][1]
    x2 = pints[1][0]
    y2 = pints[1][1]
    x3 = pints[2][0]
    y3 = pints[2][1]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    return [A, B, C]

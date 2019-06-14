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
from detectron.core.config import cfg

import pycocotools.mask as mask_util

from detectron.utils.colormap import colormap
import detectron.utils.env as envu
if cfg.MODEL.SHAPE_POINTS_ON:
    import detectron.utils.shapepoints as keypoint_utils
else:
    import detectron.utils.keypoints as keypoint_utils
import detectron.datasets.dummy_datasets as dummy_datasets

# Matplotlib requires certain adjustments in some environments
# Must happen before importing matplotlib
envu.set_up_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.misc import comb
from scipy import optimize
import math
import logging
import sys
import time
from multiprocessing import Process, Manager

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
# IMAGE_WID = 1920
# IMAGE_HEI = 1208
# SLOPE_LIMITED = IMAGE_HEI/540.#2*LANE_WID
# scale_rate = 1920./IMAGE_WID
#
# source_arr = np.float32([[918,841],[1092,841],[1103,874],[903,874]])
# source_arr = source_arr / scale_rate
# # source_arr = np.float32([[459. , 420.5],[546. , 420.5],[551.5, 437. ],[451.5, 437. ]])
# lane_wid = 200 / scale_rate
# scale_h = 0.025
# scale_w = 0.28
# offset_x = lane_wid * scale_w / 2
# offset_y = 1 - scale_h
# dest_arr = np.float32([[IMAGE_WID / 2 - offset_x, IMAGE_HEI * offset_y],
#                        [IMAGE_WID / 2  + offset_x, IMAGE_HEI * offset_y],
#                         [IMAGE_WID / 2 + offset_x, IMAGE_HEI - 1],
#                          [IMAGE_WID / 2 - offset_x, IMAGE_HEI - 1]])
# H = cv2.getPerspectiveTransform(source_arr, dest_arr)
# print ("H:" + str(H))
# print ("lane_wid:" + str(lane_wid))
#
# dist = np.array([[-0.35262804, 0.15311474, 0.00038879, 0.00048328, - 0.03534825]])
# mtx = np.array([[980.76745978, 0., 969.74796847], [0., 984.13242608, 666.25746185], [0., 0., 1.]])
# CURVETURE_MAX = 50.0/(IMAGE_HEI*IMAGE_HEI)
# manager = Manager()
# masks_list = manager.dict()

def parabola2(x, A, B, C):
    return A*x*x + B*x + C

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i



def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000


        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return np.vstack((xvals, yvals)).T


def kp_connections(keypoints):
    if cfg.MODEL.SHAPE_POINTS_ON:
        kp_lines = [
            [keypoints.index('point1'), keypoints.index('point2')],
            [keypoints.index('point2'), keypoints.index('point3')],
            [keypoints.index('point3'), keypoints.index('point4')],
        ]
    else:
        kp_lines = [
            [keypoints.index('left_eye'), keypoints.index('right_eye')],
            [keypoints.index('left_eye'), keypoints.index('nose')],
            [keypoints.index('right_eye'), keypoints.index('nose')],
            [keypoints.index('right_eye'), keypoints.index('right_ear')],
            [keypoints.index('left_eye'), keypoints.index('left_ear')],
            [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
            [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
            [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
            [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
            [keypoints.index('right_hip'), keypoints.index('right_knee')],
            [keypoints.index('right_knee'), keypoints.index('right_ankle')],
            [keypoints.index('left_hip'), keypoints.index('left_knee')],
            [keypoints.index('left_knee'), keypoints.index('left_ankle')],
            [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
            [keypoints.index('right_hip'), keypoints.index('left_hip')],
        ]
    return kp_lines


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

def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)

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


def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_one_image_opencv(
        im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
        show_box=True, dataset=None, show_class=False):
    """Constructs a numpy array with the detections visualized."""

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)
    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return im

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)
        color_list = colormap()
        mask_color_id = 0

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    im = np.array(im)

    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        # show box (off by default)
        if show_box:
            im = vis_bbox(
                im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

        # show class (off by default)
        if show_class:
            class_str = get_class_string(classes[i], score, dataset)
            im = vis_class(im, (bbox[0], bbox[1] - 2), class_str)

        # show mask
        if segms is not None and len(segms) > i:
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            im = vis_mask(im, masks[..., i], color_mask)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            im = vis_keypoints(im, keypoints[i], kp_thresh)

    return im

def get_good_parabola(coefficient):
    good_parabola = None
    good_index = 0
    min_gradient = 1
    for index, param in enumerate(coefficient):
        # point1 = get_parabola_y(param, 0)
        # point2 = get_parabola_y(param, -IMAGE_HEI)
        # point3 = get_parabola_y(param, -param[1]/(2*param[0]))
        # point3 = get_parabola_y(param, -IMAGE_HEI/2)
        # points = [point1, point2, point3]
        # dis = max(points) - min(points)

        gradient1 = get_gradient(param, -IMAGE_HEI)
        gradient2 = get_gradient(param, 0)
        gradient_dalta = abs(gradient1 - gradient2)

        if gradient_dalta < min_gradient:
            min_gradient = gradient_dalta
            good_parabola = param
            good_index = index


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

    x_array = [100, 200]
    source_p1 = [x_array[0], A * x_array[0] * x_array[0] + B * x_array[0] + C]
    theta = math.atan2(2*A*source_p1[0] + B, 1)
    point2 = [source_p1[0] + math.sin(theta) * distance, source_p1[1] + math.cos(theta) * distance]

    source_p2 = [x_array[1], A * x_array[1] * x_array[1] + B * x_array[1] + C]
    theta = math.atan2(2*A*source_p2[0] + B, 1)
    point3 = [source_p2[0] + math.sin(theta) * distance, source_p2[1] + math.cos(theta) * distance]
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

def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='pdf'):
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        print (im_name + "probability is too low, return.")
        return

    dataset_keypoints, _ = keypoint_utils.get_keypoints()

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='r',
                          linewidth=0.8, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=8,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.9, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.3)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            show_all_points = True
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if show_all_points or (kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh):
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if show_all_points or (kps[2, i1] > kp_thresh):
                    plt.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

                if show_all_points or (kps[2, i2] > kp_thresh):
                    plt.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            # mid_shoulder = (
            #     kps[:2, dataset_keypoints.index('right_shoulder')] +
            #     kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            # sc_mid_shoulder = np.minimum(
            #     kps[2, dataset_keypoints.index('right_shoulder')],
            #     kps[2, dataset_keypoints.index('left_shoulder')])
            # mid_hip = (
            #     kps[:2, dataset_keypoints.index('right_hip')] +
            #     kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            # sc_mid_hip = np.minimum(
            #     kps[2, dataset_keypoints.index('right_hip')],
            #     kps[2, dataset_keypoints.index('left_hip')])
            # if (sc_mid_shoulder > kp_thresh and
            #         kps[2, dataset_keypoints.index('nose')] > kp_thresh):
            #     x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
            #     y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
            #     line = plt.plot(x, y)
            #     plt.setp(
            #         line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            # if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            #     x = [mid_shoulder[0], mid_hip[0]]
            #     y = [mid_shoulder[1], mid_hip[1]]
            #     line = plt.plot(x, y)
            #     plt.setp(
            #         line, color=colors[len(kp_lines) + 1], linewidth=1.0,
            #         alpha=0.7)

    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
    plt.close('all')

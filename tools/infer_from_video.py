#!/usr/bin/env python2

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

"""Perform inference on a video or zmq with a certain extension
(e.g., .jpg) in a folder. Sample: 
python tools/infer_from_video.py \
--cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
--output-dir ./output \
--image-ext jpg \
--wts generalized_rcnn/model_final.pkl \
--video ~/data/video3.h264
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import sys
import time
import zmq
import numpy as np
import os

from caffe2.python import workspace
import glob
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
import arp.line_detection as detection
from multiprocessing import Process, Queue
import json
from arp.detection_filter import get_predict_list
from arp.line_detection import dist, mtx, IMAGE_WID, IMAGE_HEI, scale_size, is_px2, H_OP
CUT_OFFSET_PX2 = [277, 426]
CUT_OFFSET_PC = [302, 451]

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='png',
        type=str
    )
    parser.add_argument(
        '--video',
        help='zmq or /path/to/video/file',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


predict_time = []
process_time = []
show_img = None
def hanle_frame(args, frameId, origin_im, im, logger, model, dataset):
    global predict_time, process_time, show_img
    logger.info('Processing frame: {}'.format(frameId))
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    predict_time.append(time.time() - t)
    # logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    # logger.info('predict_time: {:.3f}s'.format(np.mean(np.array(predict_time))))
    # for k, v in timers.items():
    #     logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
    if frameId == 1:
        logger.info(
            ' \ Note: inference on the first image will be slower than the '
            'rest (caches and auto-tuning need to warm up)'
        )

    t = time.time()
    img_debug = False
    ret = detection.get_detection_line(
        im[:, :, ::-1],
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dataset,
        show_class=True,
        thresh=0.8,
        kp_thresh=2,
        frame_id=frameId,
        img_debug = img_debug
    )
    im, mid_im, top_im, result = ret
    process_time.append(time.time() - t)
    logger.info('get_detection_line time: {:.3f}s'.format(time.time() - t))
    #
    logger.info('process_time: {:.3f}s'.format(np.mean(np.array(process_time))))
    line_list = None
    cache_list = None
    if not result is None:
        line_list, cache_list = add2MsgQueue(result, frameId, img_debug)


    if img_debug:
        half_size = (int(im.shape[1]/2), int(im.shape[0]/2))
        if IMAGE_WID > 960:
            im = cv2.resize(im, half_size)
            top_im = cv2.resize(top_im, (960, 604))

            mid_im = cv2.resize(mid_im, half_size)
            # mid_im = mid_im[604:902, 0:IMAGE_WID]
            # mid_im = cv2.resize(mid_im, (int(IMAGE_WID / 2), 150))
        else:
            # mid_im = mid_im[302:451, 0:IMAGE_WID]
            pass

        if (not line_list is None) and (not cache_list is None):
            x_pos = []
            x_pos_11 = []
            prob_wid = IMAGE_WID
            if prob_wid > 960:
                prob_wid = prob_wid / 2
            for i in range(-int(prob_wid / 2), int(prob_wid / 2), 1):
                matched_y = 1
                matched_y_11 = 2
                for l in line_list:
                    dis = abs(l['x'] - i)
                    if dis < 4:
                        # hei = dis
                        if l['type'] == "boundary":
                            matched_y = int(220 * l['score'])
                        else:
                            matched_y = int(190 * l['score'] - dis * dis)
                for l in cache_list:
                    dis = abs(l['x'] - i)
                    if dis < 8:
                        matched_y_11 = int(200 * l['score'] - dis * dis)
                x_pos.append([i + int(prob_wid / 2), matched_y])
                x_pos_11.append([i + int(prob_wid / 2), matched_y_11])
            # h = np.zeros((100, IMAGE_WID, 3))
            cv2.polylines(origin_im, [np.array(x_pos)], False, (0, 255, 0))
            cv2.polylines(origin_im, [np.array(x_pos_11)], False, (0, 0, 255))
            # origin_im = np.flipud(origin_im)

            # cv2.imshow('prob', h)
            # cv2.waitKey(1)
        if not result is None:
            # for (line_param, line_type) in zip(result[0], result[1]):
            #     drawParabola(origin_im, line_param[0:3].tolist(), line_type)
            line_array = []
            for line in line_list:
                line_param = line['curve_param']
                line_type = line['type']
                points = drawParabola(origin_im, line_param[0:3], line_type)
                line_array.append(points)
            overlay = origin_im.copy()
            color = [(255,0,0), (0,255,0), (0,0,255),(255,255,0),(0,255,255),(255,0,255)]
            # for index in range(len(line_array)):
            #     if index > 0:
            #         left_line = line_array[index - 1]
            #         right_line = line_array[index]
            #         fill_points = np.array([np.append(left_line, right_line[::-1], axis=0)], dtype=np.int32)
            #         print ("fill_points:" + str(fill_points.shape))
            #         print ("color[index - 1]:" + str(color[index - 1]))
            #         cv2.fillPoly(overlay, fill_points, color[index - 1])
            # alpha = 0.2
            # cv2.addWeighted(overlay, alpha, origin_im, 1-alpha, 0, origin_im)

        # origin_im
        origin_im = np.append(origin_im, top_im, axis=1)
        im = np.append(im, mid_im, axis=1)
        show_img = np.append(origin_im, im, axis=0)
        cv2.imwrite(os.path.join(args.output_dir, "source_"+ str(frameId) + ".png"), show_img)
        cv2.imshow('carlab1', show_img)
        cv2.waitKey(1)

def drawParabola(image, line_param, type):
    points = []
    for x in range(-800, 10, 10):
        points.append([line_param[0] * x**2 + line_param[1] * x + line_param[2], x])
    points = np.array(points)
    points[:,0] = points[:,0] + IMAGE_WID/2
    points[:,1] = points[:,1] + IMAGE_HEI
    points = cv2.perspectiveTransform(np.array([points], dtype='float32'), np.array(H_OP))
    if is_px2:
        offset_y = CUT_OFFSET_PX2[0] if scale_size else 2 * CUT_OFFSET_PX2[0]
    else:
        offset_y = CUT_OFFSET_PC[0] if scale_size else 2 * CUT_OFFSET_PC[0]
    points = points[0]
    points[:,1] = points[:,1] + offset_y
    color = (0, 200, 0)
    # print ("drawParabola points:" + str(points))
    cv2.polylines(image, np.int32([np.vstack((points[:,0], points[:,1])).T]), False, color, thickness=2)
    return points

def add2MsgQueue(result, frameId, img_debug):
    line_list = []
    if (result is None) or len(result[0]) == 0:
        print ("error: len(line_list) == 0")
        return line_list, None

    for (line_param, line_type) in zip(result[0], result[1]):
        # line_info = {'curve_param':line_param[0:3].tolist(), 'type':line_type, 'score':line_param[3], 'x':line_param[4]}
        line_info = {'curve_param':line_param[0:3].tolist(), 'type':line_type, 'score':line_param[3], 'x':line_param[2]}
        line_list.append(line_info)
    line_list, cache_list = get_predict_list(line_list, frameId)

    finalMessage = {'frame': frameId, 'line_list': line_list, 'timestamp': time.time()}
    print ("finalMessage:" + str(finalMessage))
    json_str = json.dumps(finalMessage)
    if mQueue.full():
        mQueue.get_nowait()
    mQueue.put(json_str)
    return line_list, cache_list

def main(args):

    print ("main:")
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    zmq_video = args.video == "zmq"
    frameId = 0
    print ("args.video:" + str(args.video))
    socket = None
    im_list = None
    ret = None
    if zmq_video:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:6702")
    elif os.path.isdir(args.video):
        im_list = glob.glob(args.video + '/*.' + args.image_ext)
        im_list.sort()
    else:
        # From virtual camera video and its associated timestamp file on Drive PX2,e.g."./lane/videofilepath.h264"
        cap = cv2.VideoCapture(args.video)
    im_file_index = 0
    while True:
        if zmq_video:
            try:
                print ("--------------------send!")
                socket.send_string('ok')
                print ("--------------------recv!")
                message = socket.recv()
                print("Received message length:" + str(len(message)) + " type:" + str(type(message)))
                img_np = np.fromstring(message, np.uint8)
                #if len(img_np) == 0:
                #    continue
                img_np = img_np.reshape((1208, 1920,3))
                print("nparr type:" + str(type(img_np)) + " shape:" + str(img_np.shape))
                ret = True
            except KeyboardInterrupt:
                print ("interrupt received, stopping...")
                socket.close()
                context.term()
                ret = False
                cap.release()
        elif os.path.isdir(args.video):
            if im_file_index >= len(im_list):
                break
            img_np = cv2.imread(im_list[im_file_index])
            im_file_index += 1
            ret = True
            frameId += 1
        else:
            ret, img_np = cap.read()
            frameId += 1

        # read completely or raise exception
        if not ret:
            print("cannot get frame")
            break
        # if frameId < 500:
        #     continue
        if frameId % 5 == 0:
            t = time.time()
            #cv2.imwrite("tmp" + str(frameId) + ".png", img_np)
            origin_im = np.copy(img_np)
            if scale_size:
                img_np = img_np[::2]
                img_np = img_np[:,::2]
                origin_im = np.copy(img_np)
                if is_px2:
                    img_np = img_np[CUT_OFFSET_PX2[0]:CUT_OFFSET_PX2[1], 0:IMAGE_WID]
                else:
                    img_np = img_np[CUT_OFFSET_PC[0]:CUT_OFFSET_PC[1], 0:IMAGE_WID]
            else:
                origin_im = origin_im[::2]
                origin_im = origin_im[:,::2]
                if is_px2:
                    img_np = img_np[2*CUT_OFFSET_PX2[0]:2*CUT_OFFSET_PX2[1], 0:IMAGE_WID]
                else:
                    img_np = img_np[2*CUT_OFFSET_PC[0]:2*CUT_OFFSET_PC[1], 0:IMAGE_WID]
            # img_np = cv2.undistort(img_np, mtx, dist, None)
            hanle_frame(args, frameId, origin_im, img_np, logger, model, dummy_coco_dataset)
            # logger.info('hanle_frame time: {:.3f}s'.format(time.time() - t))


def result_sender():
    print ("sender process start !")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.SNDTIMEO, 3000)
    socket.bind("tcp://*:6701")
    while(True):
        message = mQueue.get(True)
        if not message is None:
            recv = socket.recv()
            print ("Received request:%s" % recv)
            try:
                socket.send(message)
            except zmq.ZMQError:
                time.sleep(1)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    mQueue = Queue(10)
    p = Process(target=result_sender)
    p.start()
    main(args)


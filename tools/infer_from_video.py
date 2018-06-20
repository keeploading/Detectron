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
--source ~/data/video3.h264
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
from multiprocessing import Process, Queue
import json

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
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--source',
        help='zmq or /path/to/video/file',
        default=None,
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def hanle_frame(args, frameId, im, logger, model, dataset):
    #out_name = os.path.join(
    #    args.output_dir, '{}'.format(frameId + '.pdf')
    #)
    logger.info('Processing frame: {}'.format(frameId))
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    for k, v in timers.items():
        logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
    if frameId == 1:
        logger.info(
            ' \ Note: inference on the first image will be slower than the '
            'rest (caches and auto-tuning need to warm up)'
        )

    # vis_utils.vis_one_image(
    #     im[:, :, ::-1],  # BGR -> RGB for visualization
    #     '{}'.format(frameId),
    #     args.output_dir,
    #     cls_boxes,
    #     cls_segms,
    #     cls_keyps,
    #     dataset=dataset,
    #     box_alpha=0.3,
    #     show_class=True,
    #     thresh=0.7,
    #     kp_thresh=2
    # )
    t = time.time()
    img_debug = True
    im, mid_im, top_im, result= vis_utils.get_detection_line(
        im[:, :, ::-1],
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dataset,
        show_class=True,
        thresh=0.7,
        kp_thresh=2,
        frame_id=frameId,
        img_debug = img_debug
    )
    logger.info('vis_one_image_opencv time: {:.3f}s'.format(time.time() - t))
    add2MsgQueue(result, frameId)


    if img_debug:
        half_size = (int(im.shape[1]/2), int(im.shape[0]/2))
        im = cv2.resize(im, half_size)
        top_im = cv2.resize(top_im, half_size)
        mid_im = mid_im[604:902, 0:1920]
        mid_im = cv2.resize(mid_im, (960, 150))
        # cv2.imwrite(os.path.join(args.output_dir, "source_"+ str(frameId) + ".png"), im)
        # cv2.imwrite(os.path.join(args.output_dir, "middle_"+ str(frameId) + ".png"), mid_im)
        # cv2.imwrite(os.path.join(args.output_dir, "top_"+ str(frameId) + ".png"), top_im)

        cv2.imshow('carlab1', im)
        cv2.imshow('carlab2', mid_im)
        cv2.imshow('carlab3', top_im)
        cv2.waitKey(1)

def add2MsgQueue(result, frameId):
    #{'frameid':id, 'line_list':[{"curve_param":[], "type":'bundary'},{}], 'timestamp':123}
    line_list = []
    for (line_param, line_type) in zip(result[0], result[1]):
        line_info = {'curve_param':line_param.tolist(), 'type':line_type}
        line_list.append(line_info)
    finalMessage = {'frame': frameId, 'line_list': line_list, 'timestamp': time.time()}
    print ("finalMessage:" + str(finalMessage))
    json_str = json.dumps(finalMessage)
    if mQueue.full():
        mQueue.get_nowait()
    mQueue.put(json_str)

def main(args):

    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    zmq_video = args.source == "zmq"
    frameId = 0

    if zmq_video:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:7001")
    else:
        # From virtual camera video and its associated timestamp file on Drive PX2,e.g."./lane/videofilepath.h264"
        cap = cv2.VideoCapture(args.source)

    while True:
        if zmq_video:
            try:
                message = socket.recv()
                print("Received message length:" + str(len(message)) + " type:" + str(type(message)))
                socket.send("ok")
                position = message.find(ZMQ_SEPER, 0, 100)
                frameId = message[:position]
                message = message[position + len(ZMQ_SEPER):]
                img_np = np.fromstring(message, np.uint8)
                img_np = img_np.reshape((400, 1400,3))
                print("nparr type:" + str(type(img_np)) + " shape:" + str(img_np.shape))
                ret = True
            except KeyboardInterrupt:
                print ("interrupt received, stopping...")
                socket.close()
                context.term()
                ret = False
                cap.release()
        else:
            ret, img_np = cap.read()
            frameId += 1

        # read completely or raise exception
        if not ret:
            print("cannot get frame")
            break
        if frameId % 20 == 0:
            t = time.time()

            # img_np = cv2.undistort(img_np, mtx, dist, None)
            hanle_frame(args, frameId, img_np, logger, model, dummy_coco_dataset)
            logger.info('hanle_frame time: {:.3f}s'.format(time.time() - t))


def result_sender():
    print ("sender process start !")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:6701")
    while(True):
        message = mQueue.get(True)
        if not message is None:
            recv = socket.recv()
            print ("Received request:%s" % recv)
            socket.send(message)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    mQueue = Queue(10)
    p = Process(target=result_sender)
    p.start()
    main(args)


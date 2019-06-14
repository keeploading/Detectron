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
import os

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--video',
        dest='video',
        help='video file',
        default='/media/administrator/deeplearning/video/video713_2.h264',
        type=str
    )
    parser.add_argument(
        '--subsample',
        dest='subsample',
        help='subsample from frames',
        default=37,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        # default='/media/administrator/deeplearning/self-labels/leftImg8bit/train/test',
        default='/media/administrator/deeplearning/self-labels/highway',
        type=str
    )
    return parser.parse_args()



def main(args):
    frameId = 0
    fileId = 0

    cap = cv2.VideoCapture(args.video)

    subsample = args.subsample
    while True:
        ret, img_np = cap.read()
        if not ret:
            print("cannot get frame")
            break
        if frameId < 0:
            continue
        frameId += 1
        if frameId % subsample == 0:
            # cv2.imshow('image', img_np)
            # k = cv2.waitKey(20)
            # if (k & 0xff == ord('q')):
            #     break
            img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img_np)
            # im = im.convert('P')
            file = "sh_{0:06}_000001_leftImg8bit.png".format(frameId)
            im.save(os.path.join(args.output_dir, file))
            print ("finish save image ---> " + file)
            fileId += 1
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    args = parse_args()
    main(args)


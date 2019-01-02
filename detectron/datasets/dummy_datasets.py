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
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.utils.collections import AttrDict
import numpy as np


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    # classes = [
    #     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    #     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    #     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    #     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    #     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    #     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    #     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    #     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    #     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    #     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    #     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    # ]
    # classes = ['__background__', 'lane']
    #
    # classes = ['__background__',
    #     'line_1',
    #     'line_2',
    #     'line_3',
    #     'line_4',
    #     'line_5',
    #     'line_6',
    #     'line_7',
    #     'line_8',
    #     'road_1',
    #     'road_12',
    #     'road_2',
    #     'road_23',
    #     'road_3',
    #     'road_34',
    #     'road_4',
    #     'road_45',
    #     'road_5',
    #     'road_56',
    #     'road_6',
    #     'road_67',
    #     'road_7',
    #     'road_78',
    #     'road_8',
    #     'car',
    #     'boundary'
    # ]
    classes = ['__background__',
        'guard rail',
        'car',
        'dashed',
        'solid',
        'solid solid',
        'dashed dashed',
        'dashed-solid',
        'solid-dashed',
        'yellow dashed',
        'yellow solid',
        'yellow solid solid',
        'yellow dashed dashed',
        'yellow dashed-solid',
        'yellow solid-dashed',
        'boundary',
        'fork_line',
        'fork_edge'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds



def get_line_dataset():
    classes = ['__background__',
        'dashed',
        'solid',
        'solid solid',
        'dashed dashed',
        'dashed-solid',
        'solid-dashed',
        'yellow dashed',
        'yellow solid',
        'yellow solid solid',
        'yellow dashed dashed',
        'yellow dashed-solid',
        'yellow solid-dashed',
        'boundary',
        'fork_line',
        'fork_edge'
    ]
    return classes

def isLaneLine(type1, type2):
    classes = [
        'dashed',
        'solid',
        'solid solid',
        'dashed dashed',
        'dashed-solid',
        'solid-dashed',
        'yellow dashed',
        'yellow solid',
        'yellow solid solid',
        'yellow dashed dashed',
        'yellow dashed-solid',
        'yellow solid-dashed',
        'fork_edge',
    ]
    return (type1 in classes and type2 in classes)

def get_color_dataset(classes):

    color_list = {'dashed':[1., 0., 0.],
                  'solid':[0., 1., 0.],
                  'boundary':[0., 0., 1.],
                  'solid solid':[1., 1., 0.],
                  'dashed dashed':[0., 1., 1.],
                  'fork_line':[1., 0., 1.],
                  'fork_edge':[1., 1., 1.]
                  }
    if not classes in color_list:
        return None
    return np.array(color_list[classes]) *255


def get_perspective_color(index):
    perspective_color = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
                         [0, 255, 255], [100, 100, 100], [100, 0, 0], [0, 100, 0], [0, 0, 100], [100, 100, 0], [100, 0, 100], [0, 100, 100],
                        [200, 200, 200], [200, 0, 0], [0, 200, 0], [0, 0, 200], [200, 200, 0], [200, 0, 200], [0, 200, 200]]
    return perspective_color[index]
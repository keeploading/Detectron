from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import h5py
import json
import os
import scipy.misc
import sys

import cityscapesscripts.evaluation.instances2dict as cs

import detectron.utils.segms as segms_util
import detectron.utils.boxes as bboxs_util
import numpy as np
import cv2
from pycocotools import mask

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset', help="cocostuff, cityscapes", default=None, type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", default=None, type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default=None, type=str)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def convert_coco_stuff_mat(data_dir, out_dir):
    """Convert to png and save json with path. This currently only contains
    the segmentation labels for objects+stuff in cocostuff - if we need to
    combine with other labels from original COCO that will be a TODO."""
    sets = ['train', 'val']
    categories = []
    json_name = 'coco_stuff_%s.json'
    ann_dict = {}
    for data_set in sets:
        file_list = os.path.join(data_dir, '%s.txt')
        images = []
        with open(file_list % data_set) as f:
            for img_id, img_name in enumerate(f):
                img_name = img_name.replace('coco', 'COCO').strip('\n')
                image = {}
                mat_file = os.path.join(
                    data_dir, 'annotations/%s.mat' % img_name)
                data = h5py.File(mat_file, 'r')
                labelMap = data.get('S')
                if len(categories) == 0:
                    labelNames = data.get('names')
                    for idx, n in enumerate(labelNames):
                        categories.append(
                            {"id": idx, "name": ''.join(chr(i) for i in data[
                                n[0]])})
                    ann_dict['categories'] = categories
                scipy.misc.imsave(
                    os.path.join(data_dir, img_name + '.png'), labelMap)
                image['width'] = labelMap.shape[0]
                image['height'] = labelMap.shape[1]
                image['file_name'] = img_name
                image['seg_file_name'] = img_name
                image['id'] = img_id
                images.append(image)
        ann_dict['images'] = images
        print("Num images: %s" % len(images))
        with open(os.path.join(out_dir, json_name % data_set), 'wb') as outfile:
            outfile.write(json.dumps(ann_dict))


# for Cityscapes
def getLabelID(self, instID):
    if (instID < 1000):
        return instID
    else:
        return int(instID / 1000)

def getAreaByPoint(points, h, w, category_id):
    line_type = 1  # cv2.CV_AA
    color = category_id
    seg = []
    for j in range(len(points)):
        coordx = points[j][0]
        coordy = points[j][1]
        point = []
        point.append(int(coordx))
        point.append(int(coordy))
        seg.append(point)

    labelMask = np.zeros((h, w))
    cv2.fillPoly(labelMask, np.array([seg], dtype=np.int32), color, line_type)

    mask_new, contours, hierarchy = cv2.findContours((labelMask).astype(np.uint8), cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)

    ##----------------------------------------------
    polygons = []
    # In practice, only one element.
    for contour in contours:
        contour = contour.flatten().tolist()
        polygons.append(contour)
    labelMask[:, :] = labelMask == color

    labelMask = np.expand_dims(labelMask, axis=2)
    labelMask = labelMask.astype('uint8')
    labelMask = np.asfortranarray(labelMask)

    Rs = mask.encode(labelMask)
    return float(mask.area(Rs))

def convert_cityscapes_instance_only(
        data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        'gtFine_train',
        'gtFine_val',
        # 'gtFine_test',

        # 'gtCoarse_train',
        # 'gtCoarse_val',
        # 'gtCoarse_train_extra'
    ]
    ann_dirs = [
        'labels_train',
        'labels_val',
        # 'gtFine_trainvaltest/gtFine/test',

        # 'gtCoarse/train',
        # 'gtCoarse/train_extra',
        # 'gtCoarse/val'
    ]
    json_name = 'instancesonly_filtered_%s.json'
    ends_in = '.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = ['__background__',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'motorcycle',
        'bicycle',
        'ground',
        'road',
        'sky'
    ]

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)
        for root, _, files in os.walk(ann_dir):
            for filename in files:
                if filename.endswith(ends_in):
                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations" % (
                            len(images), len(annotations)))
                    json_ann = json.load(open(os.path.join(root, filename)))
                    image = {}
                    image['id'] = img_id
                    img_id += 1

                    image['width'] = json_ann['imgWidth']
                    image['height'] = json_ann['imgHeight']
                    sub_name = filename.split('_')[0:3]
                    image['file_name'] = sub_name[0] + "_" + sub_name[1] + "_" + sub_name[2] + "_" + 'leftImg8bit.png'
                    images.append(image)

                    objects = json_ann["objects"]

                    for obj in objects:
                        if obj["label"] not in category_instancesonly:
                            continue  # skip non-instance categories

                        index = category_instancesonly.index(obj["label"])# + 184

                        ann = {}
                        ann['id'] = ann_id
                        ann_id += 1
                        ann['image_id'] = image['id']
                        ann['segmentation'] = [sum(obj['polygon'], [])]

                        ann['category_id'] = index
                        ann['iscrowd'] = 0

                        seg_points = obj["polygon"]
                        ann['area'] = getAreaByPoint(seg_points, image['height'], image['width'], ann['category_id'])
                        ann['bbox'] = bboxs_util.xyxy_to_xywh(
                            segms_util.polys_to_boxes(
                                [ann['segmentation']])).tolist()[0]

                        annotations.append(ann)
                    # break
        ann_dict['images'] = images
        categories = []
        for index, value in enumerate(category_instancesonly):
            categories.append({"id": index, "name": value})
        categories = categories[1:]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % data_set), 'wb') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    args.datadir = "/media/administrator/deeplearning/dataset/test_cityscape"
    args.outdir = "/media/administrator/deeplearning/dataset/test_cityscape/output"
    convert_cityscapes_instance_only(args.datadir, args.outdir)

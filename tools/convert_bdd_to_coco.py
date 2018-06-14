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

import detectron.utils.segms as segms_util
import detectron.utils.boxes as bboxs_util
from PIL import Image
import numpy as np
from scipy.misc import comb
import cv2
from pycocotools import mask


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

    return xvals, yvals


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset', help="cocostuff, cityscapes", default="cityscapes_instance_only", type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", default="output", type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted", default="input", type=str)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()

# for Cityscapes
def getLabelID(self, instID):
    if (instID < 1000):
        return instID
    else:
        return int(instID / 1000)
def getBezierPoint(polyPoint):
    xvals, yvals = bezier_curve(polyPoint, nTimes=5*len(polyPoint))
    point = []
    for x, y in zip(xvals, yvals):
        point.append([x, y])
    return point[::-1]

def getPointByPoly2d(poly2d):
    ann = []
    curve = []
    for p in poly2d:
        if p[2] == "C":
            curve.append([p[0], p[1]])
        else:
            if len(curve) > 0:
                ann.extend(getBezierPoint(curve))
                curve = []
            ann.append([p[0], p[1]])
    if poly2d[-1] == poly2d[0]:
        pass
    else:
        return []
        # np_ann = np.array(ann)
        # np_ann[:, 0] -= 5
        # np_ann = np_ann.tolist()
        # repair = np.array(ann[::-1])
        # repair[:, 0] += 5
        # repair = repair.tolist()
        # np_ann.extend(repair)
        # ann.extend(np_ann)
    return ann
def getBoxByObj(obj):
    if obj.has_key("box2d"):
        box2d = obj["box2d"]
        return [box2d["x1"], box2d["y1"],
                box2d["x2"] - box2d["x1"],
                box2d["y2"] - box2d["y1"]]
    else:
        return []

def getPointByObj(obj):
    ann = []
    box2d = []
    if obj.has_key("box2d"):
        ann.append([[obj["box2d"]["x1"], obj["box2d"]["y1"]], [obj["box2d"]["x2"], obj["box2d"]["y2"]]])
        return ann
    elif obj.has_key("poly2d"):
        area = getPointByPoly2d(obj["poly2d"])
        if len(area) > 0:
            ann.append(area)
        return ann
    elif obj.has_key("segments2d"):
        for poly in obj["segments2d"]:
            ann.append(getPointByPoly2d(poly))
        return ann

def getAreaByObj(polygon_points_array, h, w, category_id):
    line_type = 1  # cv2.CV_AA
    color = category_id
    sum = 0
    for poly_points in polygon_points_array:
        points = poly_points
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
        sum += float(mask.area(Rs))
        print ("sum:" + str(sum))
    return sum, polygons

def convert_cityscapes_instance_only(
        data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        'train',
        'val'
        # 'images/100k/train',
        # 'images/100k/val'

        # 'gtFine_train',
        # 'gtCoarse_train',
        # 'gtCoarse_val',
        # 'gtCoarse_train_extra'
    ]
    ann_dirs = [
        'annotation_train',
        'annotation_val'

        # 'labels/100k/train',
        # 'labels/100k/val'

        # 'gtFine_trainvaltest/gtFine/train',
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
        "bike",
        "bus",
        "car",
        # "motor",
        "person",
        "rider",
        "traffic light",
        "traffic sign",
        # "train",
        "truck",
        "area/alternative",
        "area/drivable",
        # "lane/crosswalk",
        # "lane/double other",
        # "lane/double white",
        # "lane/double yellow",
        # "lane/road curb",
        # "lane/single other",
        # "lane/single white",
        # "lane/single yellow"
    ]#--------------------------------------------------------------------------------------
    # Write "info"
    infodata = {'info': {'description': 'This is stable 1.0 version of the 2014 MS COCO dataset.', 'url': u'http://mscoco.org', 'version': u'1.0', 'year': 2014, 'contributor': 'Microsoft COCO group', 'date_created': '2015-01-27 09:11:52.357475'}}


    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        ann_dict["info"] = infodata["info"]
        ann_dict["type"] = 'instances'

        annPath = os.path.join(data_dir, 'coco_ref',
                               'instances_' + data_set + '2014.json')

        with open(annPath) as annFile:
            print ("open " + str(annFile))
            cocodata = json.load(annFile)
            licdata = [i for i in cocodata['licenses']]
            ann_dict["licenses"] = licdata
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

                    # im = Image.open(filename)
                    # (width, height) = im.size
                    image['width'] = 1280
                    image['height'] = 720
                    outmask = np.zeros((image['height'], image['width']), np.uint8)

                    img_dir = os.path.join(data_dir, data_set)
                    # image['file_name'] = img_dir + "/" + filename.split('.')[0] + ".jpg"
                    image['file_name'] = filename.split('.')[0] + ".jpg"
                    images.append(image)

                    # fullname = os.path.join(root, image['seg_file_name'])
                    # objects = cs.instances2dict_with_polygons(
                    #     [fullname], verbose=False)[fullname]

                    objects = json_ann["frames"][0]["objects"]
                    for obj in objects:
                        if obj["category"] not in category_instancesonly:
                            continue  # skip non-instance categories
                        index = category_instancesonly.index(obj["category"])# + 184
                        seg_points = getPointByObj(obj)#[[[point1],[point2]]]
                        seg = []
                        for seg_poit in seg_points:
                            seg.extend(sum(seg_poit, []))
                        if len(seg) == 0:
                            print('Warning: invalid segmentation.')
                            continue
                        ann = {}
                        ann['id'] = ann_id
                        ann_id += 1
                        ann['image_id'] = image['id']

                        category_dict[obj["category"]] = index
                        ann['category_id'] = index
                        ann['iscrowd'] = 0
                        if obj.has_key("box2d"):
                            ann['bbox'] = getBoxByObj(obj)
                        else:
                            ann['area'], ann['segmentation'] = getAreaByObj(seg_points, image['height'], image['width'], ann['category_id'])
                            ann['bbox'] = bboxs_util.xyxy_to_xywh(segms_util.polys_to_boxes(
                                [ann['segmentation']])).tolist()[0]

                        annotations.append(ann)
                        # break
        ann_dict['images'] = images
        # category_dict.values()
        # categories = [{"id": category_dict[name], "name": name} for name in category_dict]
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
    # args.datadir = "/media/administrator/deeplearning/dataset/bdd100k"
    # args.outdir = "/media/administrator/deeplearning/project/detectron/detectron/datasets/data/bdd/annotations"
    convert_cityscapes_instance_only(args.datadir, args.outdir)

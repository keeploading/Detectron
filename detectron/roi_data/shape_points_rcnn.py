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

"""Construct minibatches for Mask R-CNN training when keypoints are enabled.
Handles the minibatch blobs that are specific to training Mask R-CNN for
keypoint detection. Other blobs that are generic to RPN or Fast/er R-CNN are
handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from detectron.core.config import cfg
import detectron.utils.blob as blob_utils
import detectron.utils.shapepoints as keypoint_utils

logger = logging.getLogger(__name__)


def add_shape_points_rcnn_blobs(
    blobs, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx
):
    """Add Mask R-CNN keypoint specific blobs to the given blobs dictionary."""
    # Note: gt_inds must match how they're computed in
    # datasets.json_dataset._merge_proposal_boxes_into_roidb

    gt_inds = np.where(roidb['gt_classes'] > 0)[0]
    max_overlaps = roidb['max_overlaps']
    gt_keypoints = roidb['gt_shape_points']

    # print ("test11 type:", type(max_overlaps), type(gt_keypoints))

    ind_kp = gt_inds[roidb['box_to_gt_ind_map']]
    # print ("test11 max_overlaps:", max_overlaps.shape, max_overlaps)
    # print ("test11 gt_inds:", gt_inds.shape, gt_inds)
    # print ("test11 roidb['box_to_gt_ind_map']:", roidb['box_to_gt_ind_map'].shape, roidb['box_to_gt_ind_map'])
    # print ("test11 roidb['gt_classes']:", roidb['gt_classes'].shape, roidb['gt_classes'])
    # print ("test11 gt_keypoints:", gt_keypoints.shape, gt_keypoints)
    # print ("test11 roidb['boxes'].shape:", roidb['boxes'].shape)
    # print ("test11 ind_kp:", ind_kp.shape, ind_kp)
    # print ("test11 gt_classes:", roidb['gt_classes'].shape, roidb['gt_classes'] )
    # print ("test11 within_box1:", gt_keypoints[ind_kp, :, :].shape, gt_keypoints[ind_kp, :, :] )
    # print ("test11 within_box2:", roidb['boxes'].shape, roidb['boxes'] )
    within_box = _within_box(gt_keypoints[ind_kp, :, :], roidb['boxes'])
    vis_kp = gt_keypoints[ind_kp, 2, :] > 0
    # print ("test11 within_box:", within_box.shape, within_box )
    # print ("test11 vis_kp:", vis_kp.shape, vis_kp )
    is_visible = np.sum(np.logical_and(vis_kp, within_box), axis=1) > 0
    # print ("test11 is_visible:", is_visible.shape, is_visible )
    kp_fg_inds = np.where(
        np.logical_and(max_overlaps >= cfg.TRAIN.FG_THRESH, is_visible)
    )[0]
    # print ("test11 kp_fg_inds1:", kp_fg_inds.shape, kp_fg_inds )

    kp_fg_rois_per_this_image = np.minimum(fg_rois_per_image, kp_fg_inds.size)
    if kp_fg_inds.size > kp_fg_rois_per_this_image:
        kp_fg_inds = np.random.choice(
            kp_fg_inds, size=kp_fg_rois_per_this_image, replace=False
        )

    # print ("test11 kp_fg_inds2:", kp_fg_inds.shape, kp_fg_inds )
    sampled_fg_rois = roidb['boxes'][kp_fg_inds]
    # print ("test11 sampled_fg_rois:", sampled_fg_rois.shape, sampled_fg_rois )
    box_to_gt_ind_map = roidb['box_to_gt_ind_map'][kp_fg_inds]

    num_keypoints = gt_keypoints.shape[2]
    sampled_keypoints = -np.ones(
        (len(sampled_fg_rois), gt_keypoints.shape[1], num_keypoints),
        dtype=gt_keypoints.dtype
    )
    for ii in range(len(sampled_fg_rois)):
        ind = box_to_gt_ind_map[ii]
        if ind >= 0:
            # print ("test11 ind >= 0")
            sampled_keypoints[ii, :, :] = gt_keypoints[gt_inds[ind], :, :]
            assert np.sum(sampled_keypoints[ii, 2, :]) > 0

    # print ("test11 sampled_keypoints:", sampled_keypoints.shape, sampled_keypoints )
    heats, weights = keypoint_utils.keypoints_to_heatmap_labels(
        sampled_keypoints, sampled_fg_rois
    )

    shape = (sampled_fg_rois.shape[0] * cfg.KRCNN.NUM_KEYPOINTS, 1)
    heats = heats.reshape(shape)
    weights = weights.reshape(shape)

    sampled_fg_rois *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones(
        (sampled_fg_rois.shape[0], 1)
    )
    sampled_fg_rois = np.hstack((repeated_batch_idx, sampled_fg_rois))

    # print ("test11 sampled_fg_rois:", sampled_fg_rois)
    blobs['shape_points_rois'] = sampled_fg_rois
    blobs['shape_points_locations_int32'] = heats.astype(np.int32, copy=False)
    blobs['shape_points_weights'] = weights


def finalize_shape_points_minibatch(blobs, valid):
    """Finalize the minibatch after blobs for all minibatch images have been
    collated.
    """
    min_count = cfg.KRCNN.MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH
    num_visible_keypoints = np.sum(blobs['shape_points_weights'])
    valid = (
        valid and len(blobs['shape_points_weights']) > 0 and
        num_visible_keypoints > min_count
    )
    # Normalizer to use if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
    # See modeling.model_builder.add_keypoint_losses
    norm = num_visible_keypoints / (
        cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.BATCH_SIZE_PER_IM *
        cfg.TRAIN.FG_FRACTION * cfg.KRCNN.NUM_KEYPOINTS
    )
    blobs['shape_points_loss_normalizer'] = np.array(norm, dtype=np.float32)
    return valid


def _within_box(points, boxes):
    """Validate which keypoints are contained inside a given box.

    points: Nx2xK
    boxes: Nx4
    output: NxK
    """
    x_within = np.logical_and(
        points[:, 0, :] >= np.expand_dims(boxes[:, 0], axis=1),
        points[:, 0, :] <= np.expand_dims(boxes[:, 2], axis=1)
    )
    y_within = np.logical_and(
        points[:, 1, :] >= np.expand_dims(boxes[:, 1], axis=1),
        points[:, 1, :] <= np.expand_dims(boxes[:, 3], axis=1)
    )
    return np.logical_and(x_within, y_within)

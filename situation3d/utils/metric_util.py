import os
import sys
import torch
import json
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import numpy as np
from scipy.spatial.transform import Rotation as R
 
# ----------------------------------------
# Situational Localization
# ----------------------------------------

def metric_localization(
    gt_pos,
    gt_rot,
    pred_pos,
    pred_rot,
    situation_loss_tag,
):
    """
    gt_pos: [N, 3]; ground truth position, in xyz (unit is meter)
    gt_rot: [N, 4]; ground truth roation, in xyzw (quaternion)
    pred_pos: [N, 3]; predicted position, in xyz (unit is meter)
    pred_rot: [N, 4]; predicted roation, in xyzw (quaternion)
    """
    def pos_distance(pos1, pos2):
        # ignore z
        return math.sqrt(sum((pos1[:2] - pos2[:2])**2))

    def rot_distance_quat(rot1, rot2):
        # only consider rotation along z-axis, range is -pi~pi
        r1 = R.from_quat(rot1).as_rotvec()[-1]
        r2 = R.from_quat(rot2).as_rotvec()[-1]
        return min(abs(r1 - r2), 2 * math.pi - abs(r1 - r2)) / math.pi * 180

    def rot_distance_angle(rot1, rot2):
        # only consider rotation along z-axis, range is -pi~pi
        magnitude = np.sqrt(rot2[0]**2 + rot2[1]**2)
        magnitude = 1 if magnitude == 0 else magnitude
        r1 = np.arctan2(rot1[0], rot1[1])
        r2 = np.arctan2(rot2[0]/magnitude, rot2[1]/magnitude)
        return min(abs(r1 - r2), 2 * math.pi - abs(r1 - r2)) / math.pi * 180

    def convert_6d_to_rotvec(rot):
        # Assume rot is your 6D representation
        # Reshape the 6D representation back to a rotation matrix
        rotation_matrix = np.zeros((3, 3))
        rotation_matrix[:2] = rot.reshape(2, 3)
        rotation_matrix[2] = np.cross(rotation_matrix[0], rotation_matrix[1])
        rotation_matrix[2] /= np.linalg.norm(rotation_matrix[2])
        quat = R.from_matrix(rotation_matrix).as_quat()
        rotvec = R.from_quat(quat).as_rotvec()[-1]

    def rot_distance_6d(rot1, rot2):
        # Assume 6d_rot is your 6D representation
        # Reshape the 6D representation back to a rotation matrix
        r1 = convert_6d_to_rotvec(rot1)
        r2 = convert_6d_to_rotvec(rot2)
        return min(abs(r1 - r2), 2 * math.pi - abs(r1 - r2)) / math.pi * 180

    cnt_pos_0_5, cnt_pos_1 = 0, 0
    cnt_rot_15, cnt_rot_30 = 0, 0
    for gt_p, gt_r, pred_p, pred_r in zip(gt_pos, gt_rot, pred_pos, pred_rot):
        posdiff = pos_distance(gt_p, pred_p)
        if '__quat__' in situation_loss_tag:
            rotdiff = rot_distance_quat(gt_r, pred_r)
        elif '__angle__' in situation_loss_tag:
            rotdiff = rot_distance_angle(gt_r, pred_r)
        elif '__6d__' in situation_loss_tag:
            rotdiff = rot_distance_6d(gt_r, pred_r)
        else:
            raise NotImplementedError
        if posdiff < 0.5:
            cnt_pos_0_5 += 1
        if posdiff < 1:
            cnt_pos_1 += 1
        if rotdiff < 15:
            cnt_rot_15 += 1
        if rotdiff < 30:
            cnt_rot_30 += 1
    total = len(gt_pos)

    return [cnt_pos_0_5/total, 
            cnt_pos_1/total, 
            cnt_rot_15/total, 
            cnt_rot_30/total]

# ----------------------------------------
# Precision and Recall
# ----------------------------------------

def multi_scene_precision_recall(labels, pred, iou_thresh, conf_thresh, label_mask, pred_mask=None):
    '''
    Args:
        labels: (B, N, 6)
        pred: (B, M, 6)
        iou_thresh: scalar
        conf_thresh: scalar
        label_mask: (B, N,) with values in 0 or 1 to indicate which GT boxes to consider.
        pred_mask: (B, M,) with values in 0 or 1 to indicate which PRED boxes to consider.
    Returns:
        TP,FP,FN,Precision,Recall
    '''
    # Make sure the masks are not Torch tensor, otherwise the mask==1 returns uint8 array instead
    # of True/False array as in numpy
    assert(not torch.is_tensor(label_mask))
    assert(not torch.is_tensor(pred_mask))
    TP, FP, FN = 0, 0, 0
    if label_mask is None: label_mask = np.ones((labels.shape[0], labels.shape[1]))
    if pred_mask is None: pred_mask = np.ones((pred.shape[0], pred.shape[1]))
    for batch_idx in range(labels.shape[0]):
        TP_i, FP_i, FN_i = single_scene_precision_recall(labels[batch_idx, label_mask[batch_idx,:]==1, :],
                                                         pred[batch_idx, pred_mask[batch_idx,:]==1, :],
                                                         iou_thresh, conf_thresh)
        TP += TP_i
        FP += FP_i
        FN += FN_i
    
    return TP, FP, FN, precision_recall(TP, FP, FN)
      

def single_scene_precision_recall(labels, pred, iou_thresh, conf_thresh):
    """Compute P and R for predicted bounding boxes. Ignores classes!
    Args:
        labels: (N x bbox) ground-truth bounding boxes (6 dims) 
        pred: (M x (bbox + conf)) predicted bboxes with confidence and maybe classification
    Returns:
        TP, FP, FN
    """
    
    
    # for each pred box with high conf (C), compute IoU with all gt boxes. 
    # TP = number of times IoU > th ; FP = C - TP 
    # FN - number of scene objects without good match
    
    gt_bboxes = labels[:, :6]      
    
    num_scene_bboxes = gt_bboxes.shape[0]
    conf = pred[:, 6]    
        
    conf_pred_bbox = pred[np.where(conf > conf_thresh)[0], :6]
    num_conf_pred_bboxes = conf_pred_bbox.shape[0]
    
    # init an array to keep iou between generated and scene bboxes
    iou_arr = np.zeros([num_conf_pred_bboxes, num_scene_bboxes])    
    for g_idx in range(num_conf_pred_bboxes):
        for s_idx in range(num_scene_bboxes):            
            iou_arr[g_idx, s_idx] = calc_iou(conf_pred_bbox[g_idx ,:], gt_bboxes[s_idx, :])
    
    
    good_match_arr = (iou_arr >= iou_thresh)
            
    TP = good_match_arr.any(axis=1).sum()    
    FP = num_conf_pred_bboxes - TP        
    FN = num_scene_bboxes - good_match_arr.any(axis=0).sum()
    
    return TP, FP, FN
    

def precision_recall(TP, FP, FN):
    Prec = 1.0 * TP / (TP + FP) if TP+FP>0 else 0
    Rec = 1.0 * TP / (TP + FN)
    return Prec, Rec
    

def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths        
    Returns:
        iou
    """        
        
    max_a = box_a[0:3] + box_a[3:6]/2
    max_b = box_b[0:3] + box_b[3:6]/2    
    min_max = np.array([max_a, max_b]).min(0)
        
    min_a = box_a[0:3] - box_a[3:6]/2
    min_b = box_b[0:3] - box_b[3:6]/2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0*intersection / union


if __name__ == '__main__':
    print('running some tests')
    
    ############
    ## Test IoU 
    ############
    box_a = np.array([0,0,0,1,1,1])
    box_b = np.array([0,0,0,2,2,2])
    expected_iou = 1.0/8
    pred_iou = calc_iou(box_a, box_b)
    assert expected_iou == pred_iou, 'function returned wrong IoU'
    
    box_a = np.array([0,0,0,1,1,1])
    box_b = np.array([10,10,10,2,2,2])
    expected_iou = 0.0
    pred_iou = calc_iou(box_a, box_b)
    assert expected_iou == pred_iou, 'function returned wrong IoU'
    
    print('IoU test -- PASSED')
    
    #########################
    ## Test Precition Recall 
    #########################
    gt_boxes = np.array([[0,0,0,1,1,1],[3, 0, 1, 1, 10, 1]])
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0],[3, 0, 1, 1, 10, 1, 0.9]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 2 and FP == 0 and FN == 0
    assert precision_recall(TP, FP, FN) == (1, 1)
    
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 1 and FP == 0 and FN == 1
    assert precision_recall(TP, FP, FN) == (1, 0.5)
    
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0], [-1,-1,0,0.1,0.1,1, 1.0]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 1 and FP == 1 and FN == 1
    assert precision_recall(TP, FP, FN) == (0.5, 0.5)
    
    # wrong box has low confidence
    detected_boxes = np.array([[0,0,0,1,1,1, 1.0], [-1,-1,0,0.1,0.1,1, 0.1]])
    TP, FP, FN = single_scene_precision_recall(gt_boxes, detected_boxes, 0.5, 0.5)
    assert TP == 1 and FP == 0 and FN == 1
    assert precision_recall(TP, FP, FN) == (1, 0.5)
    
    print('Precition Recall test -- PASSED')
    

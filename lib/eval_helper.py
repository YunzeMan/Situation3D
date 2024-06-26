"""
Modified from: https://github.com/daveredrum/ScanRefer/blob/master/lib/eval_helper.py
"""

from re import T
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import re,sys,os
import json
from typing import List

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
from utils.metric_util import metric_localization


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx/2, sx/2, -sx/2, -sx/2, sx/2, sx/2, -sx/2, -sx/2]
    y_corners = [sy/2, -sy/2, -sy/2, sy/2, sy/2, -sy/2, -sy/2, sy/2]
    z_corners = [sz/2, sz/2, sz/2, sz/2, -sz/2, -sz/2, -sz/2, -sz/2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d


def get_eval(data_dict, config, situation_loss_tag, answer_vocab=None, use_aux_situation=False):
    """ Loss functions
    Args:
        data_dict: dict
        config: dataset config instance
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    if 'objectness_scores' in data_dict:
        objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()
        objectness_labels_batch = data_dict['objectness_label'].long()

        # construct valid mask
        pred_masks = (objectness_preds_batch == 1).float()
        label_masks = (objectness_labels_batch == 1).float()

        pred_center = data_dict['center'] # (B,K,3)
        pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal
        pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class
        pred_size_residual = pred_size_residual.squeeze(2) # B,num_proposal,3

        # store
        data_dict["pred_mask"] = pred_masks
        data_dict["label_mask"] = label_masks
        data_dict['pred_center'] = pred_center
        data_dict['pred_heading_class'] = pred_heading_class
        data_dict['pred_heading_residual'] = pred_heading_residual
        data_dict['pred_size_class'] = pred_size_class
        data_dict['pred_size_residual'] = pred_size_residual

    question_types = ['what', 'isare', 'how', 'can', 'which', 'if', 'where', 'am', 'other']
    if 'answer_scores' not in data_dict: # if using no_answer
        data_dict["answer_acc_at1"] = torch.zeros(1)[0].cuda()
        data_dict["answer_acc_at10"] = torch.zeros(1)[0].cuda()
        for i in range(len(question_types)):
            data_dict["answer_acc_breakdown_"+question_types[i]] = torch.zeros(2).cuda()
    else:
        # answer
        # data_dict['answer_scores']: batch_size, num_answers   [32, 707]
        # data_dict["answer_cats"]: batch_, num_answers         [32, 707]
        num_classes = data_dict['answer_scores'].shape[1]
        pred_answers_at1 = torch.argmax(data_dict['answer_scores'], 1) # [32]
        data_dict["answer_acc_at1"] = (F.one_hot(pred_answers_at1, num_classes=num_classes).float()
                                        * data_dict['answer_cats']).max(dim=1)[0].mean() # [32, 707] -> [32] -> [1]
        topk = 10
        pred_answers_at10 = data_dict['answer_scores'].topk(topk, dim=1)[1]
        data_dict["answer_acc_at10"] = (F.one_hot(pred_answers_at10, num_classes=num_classes).sum(dim=1).float()
                                        * data_dict['answer_cats']).max(dim=1)[0].mean()
        # question type breakdown
        # data_dict['question_type'] is a [32] tensor containing the question type index from 0 to 8
        for i in range(len(question_types)):
            question_number = (data_dict['question_type']==i).sum()
            if question_number == 0:
                data_dict["answer_acc_breakdown_"+question_types[i]] = torch.zeros(2).cuda()
            else:
                question_mask = (data_dict['question_type']==i).unsqueeze(-1).expand(-1, num_classes) # [32, 707]
                pred_answers_breakdown = F.one_hot(pred_answers_at1, num_classes=num_classes).float() * question_mask
                correct_answers = (pred_answers_breakdown*data_dict['answer_cats']).max(dim=1)[0].sum()
                data_dict["answer_acc_breakdown_"+question_types[i]] = torch.tensor([correct_answers, question_number]).cuda()

    # --------------------------------------------
    # Some other statistics
    if 'objectness_scores' in data_dict:
        obj_pred_val = torch.argmax(data_dict['objectness_scores'], 2) # B,K
        obj_acc = torch.sum((obj_pred_val==data_dict['objectness_label'].long()).float() 
                            * data_dict['objectness_mask']) / (torch.sum(data_dict['objectness_mask'])+1e-6)
        data_dict['obj_acc'] = obj_acc
        # detection semantic classification
        sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, data_dict['object_assignment']) # select (B,K) from (B,K2)
        sem_cls_pred = data_dict['sem_cls_scores'].argmax(-1) # (B,K)
        sem_match = (sem_cls_label == sem_cls_pred).float()
        data_dict["sem_acc"] = (sem_match * data_dict["pred_mask"]).sum() / data_dict["pred_mask"].sum()

    # --------------------------------------------
    # Compute situational metrics
    if use_aux_situation:
        if '__class__' in data_dict:
            acc1, acc2, acc3, acc4 = metric_localization(data_dict['auxiliary_task'][:, :3].detach().cpu().numpy(),
                                                        data_dict['auxiliary_task'][:, 3:].detach().cpu().numpy(),data_dict["aux_scores"][:, :, :1].detach().cpu().numpy(),
                                                        data_dict["aux_scores"][:, :, 1:].detach().cpu().numpy(),
                                                        situation_loss_tag)
        else:
            acc1, acc2, acc3, acc4 = metric_localization(data_dict['auxiliary_task'][:, :3].detach().cpu().numpy(),
                                                        data_dict['auxiliary_task'][:, 3:].detach().cpu().numpy(),data_dict["aux_scores"][:, :3].detach().cpu().numpy(),
                                                        data_dict["aux_scores"][:, 3:].detach().cpu().numpy(),
                                                        situation_loss_tag)
        data_dict['situation_acc_0_5m'] = torch.tensor(acc1).cuda()
        data_dict['situation_acc_1_0m'] = torch.tensor(acc2).cuda()
        data_dict['situation_acc_15deg'] = torch.tensor(acc3).cuda()
        data_dict['situation_acc_30deg'] = torch.tensor(acc4).cuda()
    else:
        data_dict['situation_acc_0_5m'] = torch.zeros(1)[0].cuda()
        data_dict['situation_acc_1_0m'] = torch.zeros(1)[0].cuda()
        data_dict['situation_acc_15deg'] = torch.zeros(1)[0].cuda()
        data_dict['situation_acc_30deg'] = torch.zeros(1)[0].cuda()


    return data_dict

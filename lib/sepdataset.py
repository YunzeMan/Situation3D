""" 
Modified from: https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py
"""

import re
import os
import sys
import time
import pickle
import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R
#from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from assets.data.scannet.model_util_scannet import ScannetDatasetConfig
sys.path.append(os.path.join(os.getcwd(), 'lib')) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from assets.data.scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes_along_axis
from utils.language_util import sqa3d_question_type
from openscene.voxelizer_dev import Voxelizer


# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, 'glove.p')

def get_index(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]

def get_answer_score(freq):
    if freq == 0:
        return .0
    else:
        return 1.

class ScannetQADatasetConfig(ScannetDatasetConfig):
    def __init__(self):
        super().__init__()
        self.num_answers = -1

class Answer(object):
    def __init__(self, answers=None, unk_token='<unk>', ignore_idx=-100):
        if answers is None:
            answers = []
        self.unk_token = unk_token
        self.ignore_idx = ignore_idx
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())
        
    def itos(self, i):
        if i == self.ignore_idx:
            return self.unk_token
        return self.rev_vocab[i]

    def stoi(self, v):
        if v not in self.vocab:
            #return self.vocab[self.unk_token]
            return self.ignore_idx
        return self.vocab[v]

    def __len__(self):
        return len(self.vocab)    


class ScannetQADataset(Dataset):
    def __init__(self, sqa, sqa_all_scene, situation_loss_tag,
            use_unanswerable=False,
            answer_cands=None,
            answer_counter=None,
            answer_cls_loss='ce',
            split='train', 
            use_color=False, 
            augment=False,
            test=False,
            use_bert=False,
        ):

        self.all_data_size = -1
        self.answerable_data_size = -1
        self.answer_features = None
        self.use_unanswerable = use_unanswerable

        # remove unanswerble qa samples for training/val
        self.all_data_size = len(sqa)
        if use_unanswerable: 
            self.sqa = sqa
        else:
            self.sqa = [data for data in sqa if len(set(data['answers']) & set(answer_cands)) > 0]
        self.answerable_data_size = len(self.sqa)
        print('all {} data: {}'.format(split, self.all_data_size))
        print('answerable {} data: {}'.format(split, self.answerable_data_size))

        self.sqa_all_scene = sqa_all_scene # all scene_ids in sqa
        self.answer_cls_loss = answer_cls_loss
        self.answer_cands = answer_cands
        self.answer_counter = answer_counter
        self.answer_vocab = Answer(answer_cands)
        self.num_answers = 0 if answer_cands is None else len(answer_cands) 

        self.split = split
        self.use_color = use_color        
        self.augment = augment
        self.test = test
        self.use_bert = use_bert
        self.situation_loss_tag = situation_loss_tag

        # tokenize a question to tokens
        scene_ids = sorted(set(record['scene_id'] for record in self.sqa))
        self.scene_id_to_number = {scene_id:int(''.join(re.sub('scene', '', scene_id).split('_'))) for scene_id in scene_ids}
        self.scene_number_to_id = {v: k for k, v in self.scene_id_to_number.items()}

        if self.use_bert:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from transformers import AutoTokenizer
            bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            def tokenize(sent):
                output = bert_tokenizer.encode_plus(sent, 
                                            add_special_tokens=True, 
                                            padding="max_length", # or "longest"
                                            max_length=CONF.TRAIN.MAX_TEXT_LEN, 
                                            truncation=True,
                                            )
                output['input_ids'] = np.array(output['input_ids'], dtype=np.int64)
                output['attention_mask'] = np.array(output['attention_mask'], dtype=np.int64)
                return output
            print('Tokenizing questions and situations using BERT Tokenizer...')
            print('This may take a while...')
            for record in self.sqa:
                record.update({'original_question': record['question'], 'original_situation': record['situation']})
                record.update(question=tokenize(record['question'])) 
                if CONF.TRAIN.USE_SITUATION:
                    record.update(situation=tokenize(record['situation']))
                else:
                    record.update(situation=tokenize(''))
            print('Finished tokenizing questions and situations using BERT Tokenizer')
        else: # use spacy tokenizer
            from spacy.tokenizer import Tokenizer
            from spacy.lang.en import English
            nlp = English()
            # Create a blank Tokenizer with just the English vocab
            spacy_tokenizer = Tokenizer(nlp.vocab)
            
            def tokenize(sent):
                sent = sent.replace('?', ' ?').replace('.', ' .')
                return [token.text for token in spacy_tokenizer(sent)]

            for record in self.sqa:
                record.update({'original_question': record['question'], 'original_situation': record['situation']})
                record.update(question=tokenize(record['question'])) 
                record.update(situation=tokenize(record['situation']))
            
        # load data
        self.voxelizer = Voxelizer(
            voxel_size=CONF.OPENSCENE.voxel_size,
            )
        self._load_data()


    def __len__(self):
        return len(self.sqa)

    def __getitem__(self, idx):
        data_dict = {}

        start = time.time()
        scene_id = self.sqa[idx]['scene_id']
        position = self.sqa[idx]['position']

        question_id = self.sqa[idx]['question_id']
        answers = self.sqa[idx].get('answers', [])
        answer_cats = np.zeros(self.num_answers) 
        answer_inds = [self.answer_vocab.stoi(answer) for answer in answers]

        if self.answer_counter is not None:        
            answer_cat_scores = np.zeros(self.num_answers)
            for answer, answer_ind in zip(answers, answer_inds):
                if answer_ind < 0:
                    continue                    
                answer_cats[answer_ind] = 1
                answer_cat_score = get_answer_score(self.answer_counter.get(answer, 0))
                answer_cat_scores[answer_ind] = answer_cat_score

            if not self.use_unanswerable:
                assert answer_cats.sum() > 0
                assert answer_cat_scores.sum() > 0
        else:
            raise NotImplementedError

        answer_cat = answer_cats.argmax()

        # get language features
        if self.use_bert:
            s_feat = self.sqa[idx]['situation']
            q_feat = self.sqa[idx]['question']
            s_len = len(s_feat['input_ids'])
            q_len = len(q_feat['input_ids'])
        else:
            s_len = self.lang[scene_id][question_id]['s_len']
            q_len = self.lang[scene_id][question_id]['q_len']
            
            s_len = s_len if s_len <= CONF.TRAIN.MAX_TEXT_LEN else CONF.TRAIN.MAX_TEXT_LEN
            q_len = q_len if q_len <= CONF.TRAIN.MAX_TEXT_LEN else CONF.TRAIN.MAX_TEXT_LEN
            s_feat = self.lang[scene_id][question_id]['s_feat']
            q_feat = self.lang[scene_id][question_id]['q_feat']
        
        # get point cloud + situational vector
        point_cloud = self.scene_data[scene_id]['locs_in_aligned'] # aligned point cloud
        pcl_color = self.scene_data[scene_id]['feats_in'] # aligned point cloud     
        labels_in = self.scene_data[scene_id]['labels_in']

        # data_dict['labels_in'] = labels_in
        # data_dict['pcl_color'] = pcl_color

        bs_center = self.scene_data[scene_id]['bs_center']
        axis_align_matrix = self.scene_data[scene_id]['axis_align_matrix']
        coord_situation = np.array(position[ : 3])
        coord_situation += bs_center # Undo the bounding sphere centering
        quat_situation = position[3 : ]
        quat_situation = np.array(quat_situation)
        augment_vector = np.ones((1, 4))
        augment_vector[:, 0 : 3] = coord_situation
        augment_vector = np.dot(augment_vector, axis_align_matrix.transpose())
        coord_situation = augment_vector[:, 0 : 3]
        coord_situation = coord_situation.reshape(-1)
        rot_situation = R.from_quat(quat_situation)
        rot_mat_situation = np.array(rot_situation.as_matrix())
        rot_mat_situation = np.dot(axis_align_matrix[0 : 3, 0 : 3], rot_mat_situation)
        rot_situation = R.from_matrix(rot_mat_situation)
        quat_situation = np.array(rot_situation.as_quat())

        # ------------------------------ DATA AUGMENTATION ------------------------------        
        if self.split == 'train' and self.augment:
            
            if not CONF.AUGMENTATION.NO_MIRROR:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                    coord_situation[0] = -1 * coord_situation[0]
                    rot_situation = R.from_quat(quat_situation).as_matrix()
                    rot_situation[0, 0] *= -1
                    rot_situation[1, 1] *= -1
                    quat_situation = list(R.from_matrix(rot_situation).as_quat())
                    
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    coord_situation[1] = -1 * coord_situation[1]
                    rot_situation = R.from_quat(quat_situation).as_matrix()
                    rot_situation = rot_situation[[1, 0, 2], :]
                    rot_situation = rot_situation[:, [1, 0, 2]]
                    quat_situation = list(R.from_matrix(rot_situation).as_quat())

            # Rotation along X-axis
            if not CONF.AUGMENTATION.NO_ROTX:
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                coord_situation = np.dot(coord_situation.reshape(1, -1), np.transpose(rot_mat)).reshape(-1)
                rot_situation = R.from_quat(quat_situation)
                rot_mat_situation = rot_situation.as_matrix()
                rot_mat_situation = np.dot(rot_mat, rot_mat_situation)
                rot_situation = R.from_matrix(rot_mat_situation)
                quat_situation = np.array(rot_situation.as_quat())

            # Rotation along Y-axis
            if not CONF.AUGMENTATION.NO_ROTY:
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                coord_situation = np.dot(coord_situation.reshape(1, -1), np.transpose(rot_mat)).reshape(-1)
                rot_situation = R.from_quat(quat_situation)
                rot_mat_situation = rot_situation.as_matrix()
                rot_mat_situation = np.dot(rot_mat, rot_mat_situation)
                rot_situation = R.from_matrix(rot_mat_situation)
                quat_situation = np.array(rot_situation.as_quat())

            # Rotation along up-axis/Z-axis
            if not CONF.AUGMENTATION.NO_ROTZ:
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                coord_situation = np.dot(coord_situation.reshape(1, -1), np.transpose(rot_mat)).reshape(-1)
                rot_situation = R.from_quat(quat_situation)
                rot_mat_situation = rot_situation.as_matrix()
                rot_mat_situation = np.dot(rot_mat, rot_mat_situation)
                rot_situation = R.from_matrix(rot_mat_situation)
                quat_situation = np.array(rot_situation.as_quat())

        # Move point clouds to the origin and update the coordinates of the situation vector
        min_coords = point_cloud.min(0)                     
        point_cloud = point_cloud - min_coords
        coord_situation = coord_situation - min_coords

        point_cloud, pcl_color, labels, inds_reconstruct, _ = self.voxelizer.voxelize(
            point_cloud, pcl_color, labels_in, return_ind=True)
        point_cloud = np.concatenate((np.ones((point_cloud.shape[0], 1), dtype=np.int64), point_cloud), axis=1)

        if '__quat__' in self.situation_loss_tag: # use quaternion
            auxiliary_task = list(coord_situation) + list(quat_situation) # 7D
        elif '__angle__' in self.situation_loss_tag: # use sin and cos of z-angle
            rotAngle = R.from_quat(quat_situation).as_rotvec()[-1]
            auxiliary_task = list(coord_situation) + [np.sin(rotAngle), np.cos(rotAngle)] # 5D
        elif '__6d__' in self.situation_loss_tag: # use 6d pose
            rot6D = np.array(R.from_quat(quat_situation).as_matrix())[:2].reshape(-1)
            auxiliary_task = list(coord_situation) + list(rot6D) # 9D
        else:
            raise NotImplementedError

        # ------------------------------- OUTPUT ------------------------------    
        # Language related inputs
        if self.use_bert:
            data_dict['s_feat'] = s_feat
            data_dict['q_feat'] = q_feat
        else:
            data_dict['s_feat'] = s_feat.astype(np.float32)
            data_dict['q_feat'] = q_feat.astype(np.float32)
        data_dict['s_len'] = np.array(s_len).astype(np.int64)
        data_dict['q_len'] = np.array(q_len).astype(np.int64)
        # 3D related inputs and labels
        data_dict['point_clouds'] = point_cloud.astype(np.int32) # np.int64 will raise error. ME doesn't take long
        if self.use_color:
            data_dict['point_colors'] = pcl_color.astype(np.float32)    # torch.Size([N, 3])
        else:
            data_dict['point_colors'] = np.ones((point_cloud.shape[0], 3), dtype=np.float32)
        data_dict['point_labels'] = labels_in.astype(np.int64)      # torch.Size([N++])
        data_dict['inds_reconstruct'] = inds_reconstruct.astype(np.int64) # torch.Size([N++])

        data_dict['scan_idx'] = np.array(idx).astype(np.int64) # torch.Size([32])
        data_dict['auxiliary_task'] = np.array(auxiliary_task).astype(np.float32) # torch.Size([32, 7/5/9])
        data_dict['scene_id'] = np.array(int(self.scene_id_to_number[scene_id])).astype(np.int64) # torch.Size([1])
        # Annotations
        if type(question_id) == str:
            data_dict['question_id'] = np.array(int(question_id.split('-')[-1])).astype(np.int64)
        else:
            data_dict['question_id'] = np.array(int(question_id)).astype(np.int64)
        data_dict['load_time'] = time.time() - start
        data_dict['answer_cat'] = np.array(int(answer_cat)).astype(np.int64) # argmax of answer_cats
        data_dict['answer_cats'] = answer_cats.astype(np.int64) # torch.Size([32, 707])
        if self.test:
            data_dict["qid"] = question_id 
        if self.answer_cls_loss == 'bce' and self.answer_counter is not None:
            data_dict['answer_cat_scores'] = answer_cat_scores.astype(np.float32) # num_answers
        # Here we add break down of the question type
        question_type = sqa3d_question_type(self.sqa[idx]['original_question'])
        data_dict['question_type'] = np.array(question_type).astype(np.int64)
        data_dict['situation'] = self.sqa[idx]['original_situation']
        data_dict['question'] = self.sqa[idx]['original_question']
        data_dict['answers'] = self.sqa[idx]['answers']

        return data_dict

    def _tranform_text_glove(self, token_type='token'):
        with open(GLOVE_PICKLE, 'rb') as f:
            glove = pickle.load(f)

        lang = {}
        for data in self.sqa:
            scene_id = data['scene_id']
            question_id = data['question_id']

            if scene_id not in lang:
                lang[scene_id] = {}

            if question_id in lang[scene_id]:
                continue
            lang[scene_id][question_id] = {}
            # tokenize the description
            s_tokens = data["situation"]
            q_tokens = data["question"]
            s_embeddings = np.zeros((CONF.TRAIN.MAX_TEXT_LEN, 300))
            q_embeddings = np.zeros((CONF.TRAIN.MAX_TEXT_LEN, 300))
            for token_id in range(CONF.TRAIN.MAX_TEXT_LEN):
                if token_id < len(s_tokens):
                    token = s_tokens[token_id]
                    if CONF.TRAIN.USE_SITUATION:     
                        if token in glove:
                            s_embeddings[token_id] = glove[token]
                        else:
                            s_embeddings[token_id] = glove['unk']
                    else:
                        s_embeddings[token_id] = glove['unk']
            for token_id in range(CONF.TRAIN.MAX_TEXT_LEN):
                if token_id < len(q_tokens):
                    token = q_tokens[token_id]
                    if token in glove:
                        q_embeddings[token_id] = glove[token]
                    else:
                        q_embeddings[token_id] = glove['unk']
            # store
            lang[scene_id][question_id]['s_feat'] = s_embeddings
            lang[scene_id][question_id]['s_len'] = len(s_tokens)
            lang[scene_id][question_id]['q_feat'] = q_embeddings
            lang[scene_id][question_id]['q_len'] = len(q_tokens)
            lang[scene_id][question_id]['s_token'] = s_tokens
            lang[scene_id][question_id]['q_token'] = q_tokens
        temp = list(DC.type2class.keys())
        class_embedding = np.zeros((len(temp), 300))
        for token_id in range(len(temp)):
            token = temp[token_id]
            if token in glove:
                class_embedding[token_id] = glove[token]
            else:
                class_embedding[token_id] = glove['unk']
        self.class_embedding = class_embedding
        return lang

    def _load_data(self):
        print('loading data...')
        # load language features
        if self.use_bert:
            pass # already tokenized
        else:
            self.lang = self._tranform_text_glove('token')

        # add scannet data
        self.scene_list = sorted(list(set([data['scene_id'] for data in self.sqa])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # ======= Load Meta Data =======
            meta_file = open(os.path.join(CONF.PATH.SCANNET_SCANS, scene_id, scene_id+".txt")).readlines()
            axis_align_matrix = None
            for line in meta_file:
                if 'axisAlignment' in line:
                    axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            if axis_align_matrix != None:
                axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
            self.scene_data[scene_id]['axis_align_matrix'] = axis_align_matrix if axis_align_matrix is not None else np.eye(4)

            # ======= Load Openscene 3D Data (ScanNet) =======
            file_train_path = os.path.join(CONF.OPENSCENE.data_root, 'train', scene_id)+'_vh_clean_2.pth'
            file_val_path = os.path.join(CONF.OPENSCENE.data_root, 'val', scene_id)+'_vh_clean_2.pth'
            locs_in, feats_in, labels_in = torch.load(file_train_path) if os.path.exists(file_train_path) else torch.load(file_val_path)
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            feats_in = (feats_in + 1.) * 127.5

            self.scene_data[scene_id]['bs_center'] = (np.max(locs_in, axis=0) + np.min(locs_in, axis=0)) / 2

            # convert to axis-aligned coordinates
            pts = np.ones((locs_in.shape[0], 4))
            pts[:,0:3] = locs_in[:,0:3]
            pts = pts @ axis_align_matrix.transpose() # Nx4
            self.scene_data[scene_id]['locs_in_aligned'] = pts[:,0:3]
            self.scene_data[scene_id]['feats_in'] = feats_in
            self.scene_data[scene_id]['labels_in'] = labels_in


def SQA3D_collate_fn(batch_list):
    """
    Custom collate function that handles specific batching for 'point_clouds', 'point_colors', 
    'point_labels', and 'inds_reconstruct' keys in the data dictionary,
    and uses default_collate for all other keys.
    """
    # Extract points and colors
    points_list = [d['point_clouds'] for d in batch_list]
    colors_list = [d['point_colors'] for d in batch_list]
    labels_list = [d['point_labels'] for d in batch_list]
    inds_reconstruct_list = [d['inds_reconstruct'] for d in batch_list]

    # Implement custom batching strategy for 
    # 'point_clouds', 'point_colors', 'point_labels', 'inds_reconstruct' here
    accmulate_points_num = 0
    for i in range(len(points_list)):
        points_list[i][:, 0] *= i
        inds_reconstruct_list[i] = accmulate_points_num + inds_reconstruct_list[i]
        accmulate_points_num += points_list[i].shape[0]

    # Use default_collate for other keys
    # First, remove 'points' and 'colors' keys from the dictionaries
    for d in batch_list:
        del d['point_clouds']
        del d['point_colors']
        del d['point_labels']
        del d['inds_reconstruct']
    batched_data = default_collate(batch_list)
    
    # Add batched_points and batched_colors to batched_data
    batched_data['point_clouds'] = torch.from_numpy(np.concatenate(points_list, axis=0))
    batched_data['point_colors'] = torch.from_numpy(np.concatenate(colors_list, axis=0))
    batched_data['point_labels'] = torch.from_numpy(np.concatenate(labels_list, axis=0))
    batched_data['inds_reconstruct'] = torch.from_numpy(np.concatenate(inds_reconstruct_list, axis=0))

    return batched_data

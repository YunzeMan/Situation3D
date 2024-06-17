import os
import sys
import json
import argparse
import collections
import torch
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
from torch.utils.data import DataLoader
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from torch.utils import model_zoo
from easydict import EasyDict
from open3d.visualization import rendering

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
sys.path.append(os.path.join(os.path.dirname(os.getcwd()))) # HACK add the parent folder
from lib.sepdataset import ScannetQADataset, ScannetQADatasetConfig, SQA3D_collate_fn
from lib.solver import Solver
from lib.config import CONF
from models.sqa_module import SIG3D
from collections import OrderedDict
import transformers
from utils.visualization import visualize_scene_test, visualize_scene_test_good_res_video
import MinkowskiEngine as ME
from utils import segmentation_util
import tqdm 

# constants
DC = ScannetQADatasetConfig()

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. XYZ_COLOR", default="")
    parser.add_argument("--tags", type=str, help="tags for the wandb, separate with _, e.g. default_color_aux", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    ## Situation
    parser.add_argument("--situation_loss_tag", type=str, default="__l2__quat__", help="situation loss type")
    # Training
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--train_num_scenes", type=int, default=-1, help="Number of train scenes [default: -1]")
    parser.add_argument("--val_num_scenes", type=int, default=-1, help="Number of val scenes [default: -1]")
    parser.add_argument("--test_num_scenes", type=int, default=-1, help="Number of test scenes [default -1]")

    parser.add_argument("--use_bert", action="store_true", help="Use bert as language encoder.") # 
    parser.add_argument("--bert_model_name", type=str, help="Pretrained bert model name", default='sentence-transformers/all-mpnet-base-v2') # or bert-base-uncased, bert-large-uncased-whole-word-masking, distilbert-base-uncased
    parser.add_argument("--freeze_bert", action="store_true", help="Freeze bert parameters.")
    parser.add_argument("--finetune_bert_last_layer", action="store_true", help="Finetune bert last layer.")
    parser.add_argument("--finetune_bert_full", action="store_true", help="Finetune bert all layers.")
    # Data
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    # Model
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden layer size[default: 256]")
    ## module option
    parser.add_argument("--no_object_mask", action="store_true", help="objectness_mask for qa")
    parser.add_argument("--no_answer", action="store_true", help="Do NOT train the localization module.")
    # Pretrain
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    # Answer
    parser.add_argument("--answer_cls_loss", type=str, help="answer classifier loss", default="bce") # ce, bce
    parser.add_argument("--answer_max_size", type=int, help="maximum size of answer candicates", default=-1) # default use all
    parser.add_argument("--answer_min_freq", type=int, help="minimum frequence of answers", default=1)
    parser.add_argument("--answer_pdrop", type=float, help="dropout_rate of answer_cls", default=0.3)
    # Question
    parser.add_argument("--lang_num_layers", type=int, default=1, help="Number of GRU layers")
    parser.add_argument("--lang_use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--lang_pdrop", type=float, help="dropout_rate of lang_cls", default=0.3)
    ## MCAN
    parser.add_argument("--mcan_pdrop", type=float, help="", default=0.1)
    parser.add_argument("--mcan_flat_mlp_size", type=int, help="", default=256) # mcan: 512
    parser.add_argument("--mcan_flat_glimpses", type=int, help="", default=1)
    parser.add_argument("--mcan_flat_out_size", type=int, help="", default=512) # mcan: 1024
    parser.add_argument("--mcan_num_heads", type=int, help="", default=8)
    parser.add_argument("--mcan_num_layers", type=int, help="", default=2) # mcan: 6
    ## which split to evaluate
    parser.add_argument("--split", type=str, choices=['train', 'val', 'test'], default='train')
    ## checkpoint
    parser.add_argument("--ckpt", type=str, help="checkpoint to evaluate")
    parser.add_argument("--folder_name", type=str, default="openscene")
    args = parser.parse_args()
    return args


def get_answer_cands(args, answer_counter_list):
    answer_counter = answer_counter_list
    answer_counter = collections.Counter(sorted(answer_counter))
    num_all_answers = len(answer_counter)
    answer_max_size = args.answer_max_size
    if answer_max_size < 0:
        answer_max_size = len(answer_counter)
    answer_counter = dict([x for x in answer_counter.most_common()[:answer_max_size] if x[1] >= args.answer_min_freq])
    print("using {} answers out of {} ones".format(len(answer_counter), num_all_answers))
    answer_cands = sorted(answer_counter.keys())
    return answer_cands, answer_counter


def get_dataloader(args, sqa, all_scene_list, split, config, augment, answer_counter_list, test=False):
    answer_cands, answer_counter = get_answer_cands(args, answer_counter_list)
    config.num_answers = len(answer_cands)

    tokenizer = None

    dataset = ScannetQADataset(
        sqa=sqa[split],
        sqa_all_scene=all_scene_list,
        situation_loss_tag=args.situation_loss_tag,
        answer_cands=answer_cands,
        answer_counter=answer_counter,
        answer_cls_loss=args.answer_cls_loss,
        split=split,
        use_color=args.use_color,
        augment=augment,
        use_bert=args.use_bert,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, collate_fn=SQA3D_collate_fn)
    return dataset, dataloader


def get_model(args, config):
    lang_emb_size = 300 # glove emb_size

    model = SIG3D(
        num_answers=config.num_answers,
        # situation
        situation_loss_tag=args.situation_loss_tag,
        # qa
        answer_pdrop=args.answer_pdrop,
        mcan_num_layers=args.mcan_num_layers,
        mcan_num_heads=args.mcan_num_heads,
        mcan_pdrop=args.mcan_pdrop,
        mcan_flat_mlp_size=args.mcan_flat_mlp_size,
        mcan_flat_glimpses=args.mcan_flat_glimpses,
        mcan_flat_out_size=args.mcan_flat_out_size,
        # lang
        lang_use_bidir=args.lang_use_bidir,
        lang_num_layers=args.lang_num_layers,
        lang_emb_size=lang_emb_size,
        lang_pdrop=args.lang_pdrop,
        # common
        hidden_size=args.hidden_size,
        # option
        use_answer=(not args.no_answer),
        use_bert=args.use_bert,
        bert_model_name=args.bert_model_name,
        freeze_bert=args.freeze_bert,
        finetune_bert_last_layer=args.finetune_bert_last_layer,
        finetune_bert_full=args.finetune_bert_full,
    )

    # to CUDA
    model = model.cuda()
    print(next(model.parameters()).device)
    return model

def get_sqa(sqa_train, sqa_val, sqa_test, train_num_scenes, val_num_scenes, test_num_scenes):
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in sqa_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in sqa_val])))
    test_scene_list = sorted(list(set([data["scene_id"] for data in sqa_test])))
    # set train_num_scenes
    if train_num_scenes <= -1:
        train_num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= train_num_scenes

    # slice train_scene_list
    train_scene_list = train_scene_list[:train_num_scenes]

    # filter data in chosen scenes
    new_sqa_train = []
    for data in sqa_train:
        if data["scene_id"] in train_scene_list:
            new_sqa_train.append(data)

    # set val_num_scenes
    if val_num_scenes <= -1:
        val_num_scenes = len(val_scene_list)
    else:
        assert len(val_scene_list) >= val_num_scenes

    # slice val_scene_list
    val_scene_list = val_scene_list[:val_num_scenes]

    new_sqa_val = []
    for data in sqa_val:
        if data["scene_id"] in val_scene_list:
            new_sqa_val.append(data)

    # set val_num_scenes
    if test_num_scenes <= -1:
        test_num_scenes = len(test_scene_list)
    else:
        assert len(test_scene_list) >= test_num_scenes

    # slice val_scene_list
    test_scene_list = test_scene_list[:test_num_scenes]

    new_sqa_test = []
    for data in sqa_test:
        if data["scene_id"] in test_scene_list:
            new_sqa_test.append(data)

    # all sqa scene
    all_scene_list = train_scene_list + val_scene_list + test_scene_list
    return new_sqa_train, new_sqa_val, new_sqa_test, all_scene_list

def test(args, SQA_TRAIN, SQA_VAL, SQA_TEST, path, answer_counter_list):

    sqa_train, sqa_val, sqa_test, all_scene_list = get_sqa(SQA_TRAIN, SQA_VAL, SQA_TEST, args.train_num_scenes, args.val_num_scenes, args.test_num_scenes)
    sqa = {
        "train" : sqa_train,
        "val" : sqa_val,
        "test" : sqa_test
    }
    val_dataset, val_dataloader = get_dataloader(args, sqa, all_scene_list, args.split, DC, False, answer_counter_list, test=True)

    ckpt = torch.load(path, map_location='cuda:{}'.format(args.gpu))
    model = get_model(args, DC)
    sd_before_load = deepcopy(model.state_dict())
    model.load_state_dict(ckpt, strict=False) # strict=True strict=False
    sd_after_load = deepcopy(model.state_dict())
    same_keys = [k for k in sd_before_load if torch.equal(sd_before_load[k], sd_after_load[k])]
    new_keys = []
    for key in same_keys:
        new_keys.append(key)
    print('-------------------- Loaded weights --------------------')
    print(f'Weights unloaded:{new_keys}')
    print('----------------------------')
    
    # precompute text features
    text_features, labelset, mapper, palette = \
        segmentation_util.precompute_text_related_properties('scannet_3d', CONF.OPENSCENE.feature_2d_extractor)


    visualization_root = './visualization/{}'.format(args.folder_name)
    os.makedirs(visualization_root, exist_ok=True)

    scene_number_to_id = val_dataloader.dataset.scene_number_to_id
    model.to('cuda:{}'.format(args.gpu))
    model.eval()

    # setup open3d renderer
    # NOTE: To show the real color in open3d, set_post_processing(False), and use DefaultUnlit shader
    w, h = 640, 480
    render = rendering.OffscreenRenderer(w, h)
    # render.scene.set_background([1.0, 1.0, 1.0, 0.0])
    render.scene.view.set_post_processing(False)
    render.scene.show_axes(False)
    render.scene.scene.set_sun_light([-1, -1, -1], [2.0, 2.0, 2.0], 100000)
    render.scene.scene.enable_sun_light(True)
    aspect = h/w
    s = 1.25
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho, -s, s, -s*aspect, s*aspect, 0.1, 200)

    # set random seed
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)

    VIS_CONF = EasyDict()
    VIS_CONF.MAX_COUNT = 200
    VIS_CONF.GT_SVEC = True
    VIS_CONF.PRED_SVEC = False
    VIS_CONF.ACTIVATION = False 
    VIS_CONF.POINT = False
    VIS_CONF.LOOK_AT = [[0, 0, 0], [0, 0, 50], [0, 0, 1]] # look_at(center, eye, up)
    VIS_CONF.VIDEO = False
    VIS_CONF.GENERATE_MINUS = False
    # VIS_CONF.LOOK_AT = [[0, 0, 0], [40, 0, 50], [0, 0, 1]]
    with torch.no_grad():
        count, correct_count = 0, 0
        preds_segmentation, gts_segmentation = [], []
        for data_dict in tqdm.tqdm(val_dataloader):
            if count >= VIS_CONF.MAX_COUNT: break
            for key in data_dict:
                if type(data_dict[key]) is dict or isinstance(data_dict[key], transformers.tokenization_utils_base.BatchEncoding):
                    data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
                elif isinstance(data_dict[key], list):
                    data_dict[key] = data_dict[key]
                else:
                    data_dict[key] = data_dict[key].cuda()

            # process points into sparse tensor and forward pass
            pcl = data_dict['point_clouds']
            feat = data_dict['point_colors']
            data_dict['openscene_in'] = ME.SparseTensor(feat.cuda(non_blocking=True), pcl.cuda(non_blocking=True))
            data_dict = model(data_dict)

            # get segmentation prediction
            point_feat = data_dict['openscene_out']
            inds_reverse = data_dict['inds_reconstruct']
            point_feat = point_feat[inds_reverse, :]
            pred = point_feat.half() @ text_features.t()
            logits_pred = torch.max(pred, 1)[1].cpu()
            gts_label = data_dict['point_labels'].cpu()
            preds_segmentation.append(logits_pred)
            gts_segmentation.append(gts_label)
                
            # Visualize the attention map with 3D scene
            if count < VIS_CONF.MAX_COUNT:
                att_feat_ori = data_dict["att_feat_ori"][0].cpu()  # att_feat_ori  att_feat_pre
                points_position = data_dict["scene_positions"][0].cpu()
                feats = [att_feat_ori]
                feats_name = ['PRE'] # A_after_cross_attention  
                scene_id = scene_number_to_id[data_dict['scene_id'].tolist()[0]]
                for feat, feat_name in zip(feats, feats_name):
                    feat -= feat.mean(dim=0, keepdim=True)
                    activation = feat.norm(dim=-1).numpy()
                    activation = (activation - activation.min()) / (activation.max() - activation.min())

                    # visualize_scene_test(data_dict, scene_id, args.situation_loss_tag, visualization_root, count, activation, feat_name, points_position.clone(), VIS_CONF, render)
                    visualize_scene_test_good_res_video(data_dict, scene_id, args.situation_loss_tag, visualization_root, count, activation, feat_name, points_position.clone(), VIS_CONF, render)
            
            # get QA prediction
            pred_answer = torch.argmax(data_dict["answer_scores"], 1).cpu().detach().item()
            gt_answer = torch.argmax(data_dict["answer_cats"].squeeze()).cpu().detach().item()
            count += 1
            correct_count = correct_count + 1 if pred_answer == gt_answer else correct_count

            
        # calculate and print QA accuracy
        print("overall QA acc:", correct_count / count)
        
        # calculate and print segmentation accuracy
        gt_segmentation = torch.cat(gts_segmentation)
        pred_segmentation = torch.cat(preds_segmentation)
        current_iou = segmentation_util.evaluate(pred_segmentation.numpy(),
                                                    gt_segmentation.numpy(),
                                                    dataset='scannet_3d',
                                                    stdout=True)

    return correct_count / count

if __name__ == "__main__":
    args = parse_option()
    # setting
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    project_name = "SQA"
    SQA_TRAIN = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_train.json")))
    SQA_VAL = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_test.json")))
    SQA_TEST = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_test.json")))
    answer_counter_list = json.load(open(os.path.join(CONF.PATH.SQA, "answer_counter.json")))
    torch.cuda.set_device('cuda:{}'.format(args.gpu))
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # reproducibility
    torch.manual_seed(CONF.TRAIN.SEED)
    torch.cuda.manual_seed(CONF.TRAIN.SEED)
    torch.cuda.manual_seed_all(CONF.TRAIN.SEED)
    random.seed(CONF.TRAIN.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(CONF.TRAIN.SEED)
    path = args.ckpt
    save_list = test(args, SQA_TRAIN, SQA_VAL, SQA_TEST, path, answer_counter_list)

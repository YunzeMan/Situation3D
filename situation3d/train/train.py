import os
import sys
import json
import argparse
import collections
import torch
import torch.optim as optim
import numpy as np
import wandb
import time 
from tensorboardX import SummaryWriter 
import random
from torch.utils.data import DataLoader
from torch.utils import model_zoo 
from datetime import datetime

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
sys.path.append(os.path.join(os.path.dirname(os.getcwd()))) # HACK add the parent folder
from lib.solver import Solver
from lib.config import CONF
from lib.sepdataset import ScannetQADataset, ScannetQADatasetConfig, SQA3D_collate_fn
from models.sqa_module import SIG3D
from utils.count_parameters import count_parameters

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
    parser.add_argument("--cur_criterion", type=str, default="answer_acc_at1")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=40)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=1000) # 5000
    parser.add_argument("--train_num_scenes", type=int, default=-1, help="Number of train scenes [default: -1]")
    parser.add_argument("--val_num_scenes", type=int, default=-1, help="Number of val scenes [default: -1]")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers for data loading")
    parser.add_argument("--test_num_scenes", type=int, default=-1, help="Number of test scenes [default -1]")
    # Optimizer
    parser.add_argument("--optim_name", type=str, help="optimizer name", default="adam")
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-1)
    parser.add_argument("--lr", type=float, help="initial learning rate", default=5e-5)
    parser.add_argument("--adam_beta1", type=float, help="beta1 hyperparameter for the Adam optimizer", default=0.9)
    parser.add_argument("--adam_beta2", type=float, help="beta2 hyperparameter for the Adam optimizer", default=0.999) # 0.98
    parser.add_argument("--adam_epsilon", type=float, help="epsilon hyperparameter for the Adam optimizer", default=1e-8) # 1e-9
    parser.add_argument("--amsgrad", action="store_true", help="Use amsgrad for Adam")
    parser.add_argument("--lr_scheduler_type", type=str, help="lr scheduler type", default=None) # step, cos
    parser.add_argument('--lr_decay_step', nargs='+', type=int, default=[30, 40]) # 15
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of learning rate", default=0.1) # 0.1, 0.2
    parser.add_argument('--bn_decay_step', type=int, default=20)
    parser.add_argument("--bn_decay_rate", type=float, help="bn rate", default=0.5)
    parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm ", default=1.0)
    parser.add_argument("--sep_params", action="store_true", help="Separate parameters in BERT and other modules.")
    parser.add_argument("--lrBERT", type=float, help="initial learning rate for BERT", default=5e-5)
    parser.add_argument("--lrLLAMA", type=float, help="initial learning rate for BERT", default=2e-4)
    parser.add_argument("--lrOther", type=float, help="initial learning rate for other modules", default=5e-4)

    parser.add_argument("--use_bert", action="store_true", help="Use bert as language encoder.") # 
    parser.add_argument("--bert_model_name", type=str, help="Pretrained bert model name", default='sentence-transformers/all-mpnet-base-v2') # or bert-base-uncased, bert-large-uncased-whole-word-masking, distilbert-base-uncased
    parser.add_argument("--freeze_bert", action="store_true", help="Freeze bert parameters.")
    parser.add_argument("--finetune_bert_last_layer", action="store_true", help="Finetune bert last layer.")
    parser.add_argument("--finetune_bert_full", action="store_true", help="Finetune bert all layers.")
    # Data
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    # Model
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden layer size[default: 768]")
    ## module option
    parser.add_argument("--no_object_mask", action="store_true", help="objectness_mask for qa")
    parser.add_argument("--no_answer", action="store_true", help="Do NOT train the QA module.")
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


def get_dataloader(args, sqa, all_scene_list, split, config, augment, answer_counter_list):
    answer_cands, answer_counter = get_answer_cands(args, answer_counter_list)
    config.num_answers = len(answer_cands)

    dataset = ScannetQADataset(
        sqa=sqa[split], # sqa[split][:100] for debug,
        sqa_all_scene=all_scene_list,
        situation_loss_tag=args.situation_loss_tag, # loss type, e.g. __l2__quat__
        answer_cands=answer_cands,
        answer_counter=answer_counter,
        answer_cls_loss=args.answer_cls_loss,
        split=split,
        use_color=args.use_color, # remember to set to False
        augment=augment, 
        use_bert=args.use_bert,
        # situation
    )

    if split == "train":
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=SQA3D_collate_fn)
    elif split == "val":
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=SQA3D_collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=SQA3D_collate_fn)
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
        hidden_size=args.hidden_size, # 768
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


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, dataloader):
    model = get_model(args, DC)
    count_parameters(model) # print number of parameters
    
    no_decay_filter = ["bias", "LayerNorm.weight"]
    no_update_filter = CONF.TRAIN.no_update_filter # ["openscene_net"]
    if args.sep_params: # NOTE: DO not use this, no LLAMA is used
        LLAMA_decay, LLAMA_no_decay, Others_decay, Others_no_decay = [], [], [], []
        for name, param in model.named_parameters():
            # if name does not contain 'voting_net' or 'proposal_net' or 'detection_backbone', then use BERT decay
            if not param.requires_grad:
                continue # frozen weights
            if any(nd in name for nd in no_update_filter):
                continue # skip llama.layers
            if 'llama_dim_mapper1' in name or 'llama_dim_mapper2' in name:
                if not any(nd in name for nd in no_decay_filter): 
                    LLAMA_decay.append(param)
                else:
                    LLAMA_no_decay.append(param)
            else:
                if not any(nd in name for nd in no_decay_filter):
                    Others_decay.append(param)
                else:
                    Others_no_decay.append(param)
        params = [
            {'params': LLAMA_decay, 'weight_decay': args.wd, 'lr': args.lrLLAMA},
            {'params': LLAMA_no_decay, 'weight_decay': 0.0, 'lr': args.lrLLAMA},
            {'params': Others_decay, 'weight_decay': args.wd, 'lr': args.lrBERT},
            {'params': Others_no_decay, 'weight_decay': 0.0, 'lr': args.lrBERT},]
    else:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue # frozen weights
            if any(nd in name for nd in no_update_filter):
                continue # skip llama.layers
            if not any(nd in name for nd in no_decay_filter):
                decay.append(param)
            else:
                no_decay.append(param)
        params = [
            {'params': decay, 'weight_decay': args.wd},
            {'params': no_decay, 'weight_decay': 0.0},]

    if args.optim_name == 'adam':
        optimizer = optim.Adam(
            params,
            lr=args.lr,
            betas=[args.adam_beta1, args.adam_beta2],
            eps=args.adam_epsilon,
            amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw':
        optimizer = optim.AdamW(params,
                                lr=args.lr,
                                betas=[args.adam_beta1, args.adam_beta2],
                                eps=args.adam_epsilon,
                                amsgrad=args.amsgrad)
    elif args.optim_name == 'adamw_cb':
        from transformers import AdamW
        optimizer = AdamW(params, 
                            lr=args.lr,
                            betas=[args.adam_beta1, args.adam_beta2],
                            eps=args.adam_epsilon)
    else:
        raise NotImplementedError

    print('set optimizer...')
    print(optimizer)
    print()
    
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.tag: stamp += "_"+args.tag.upper()
    # CONF.PROJECTID = stamp
    
    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)
    
    if 'llama' in args.tags:
        print("Loading LLaMA checkpoints")
        start_time = time.time()
        # checkpoints = sorted(Path(args.llama_path).glob("*.pth"))
        # ckpt_path = checkpoints[0]
        ckpt_path = '/scratch/bbjv/ziqip2/LLaMA/7B/consolidated.00.pth'
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cuda'))
        model.llama.custom_load_state_dict(checkpoint, tail=True, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds") 
    if 'openscene' in args.tags:
        print("Loading OpenScene checkpoints")
        checkpoint = model_zoo.load_url(CONF.OPENSCENE.model_url, progress=True)
        model.openscene_net.load_state_dict(checkpoint['state_dict'], strict=True)

    tags = ['default']
    if args.tags: tags = args.tags.split('_')
    
    os.makedirs(os.path.join(CONF.PATH.OUTPUT, 'wandb'), exist_ok=True)
    wandb.login()
    wandb.init(project='sqa3d-dev', 
               id=stamp, 
               tags=tags, 
               dir=os.path.join(CONF.PATH.OUTPUT, 'wandb'))

    wandb.watch(model, log='all', log_freq=100)
    
    solver = Solver(
        args=args,
        model=model,
        config=DC,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        cur_criterion=args.cur_criterion,
        use_answer=not args.no_answer,
        max_grad_norm=args.max_grad_norm,
        lr_decay_step=args.lr_decay_step,
        lr_decay_rate=args.lr_decay_rate,
        bn_decay_step=args.bn_decay_step,
        bn_decay_rate=args.bn_decay_rate,
        situation_loss_tag=args.situation_loss_tag,
    )
    num_params = get_num_params(model)

    return solver, num_params, root, stamp


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params
    
    wandb.config.update(info)

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    answer_vocab = train_dataset.answer_counter
    with open(os.path.join(root, "answer_vocab.json"), "w") as f:
        json.dump(answer_vocab, f, indent=4)



def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list


def get_sqa(sqa_train, sqa_val, train_num_scenes, val_num_scenes):
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in sqa_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in sqa_val])))
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

    # all sqa scene
    all_scene_list = train_scene_list + val_scene_list
    return new_sqa_train, new_sqa_val, all_scene_list


def train(args, SQA_TRAIN, SQA_VAL, answer_counter_list):

    # init training dataset
    print("preparing data...")
    sqa_train, sqa_val, all_scene_list = get_sqa(SQA_TRAIN, SQA_VAL, args.train_num_scenes, args.val_num_scenes)
    sqa = {
        "train": sqa_train,
        "val": sqa_val,
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, sqa, all_scene_list, "train", DC, CONF.TRAIN.USE_AUGMENTATION, answer_counter_list)
    val_dataset, val_dataloader = get_dataloader(args, sqa, all_scene_list, "val", DC, False, answer_counter_list)
    print("train on {} samples and val on {} samples".format(len(train_dataset), len(val_dataset)))

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }
        
    print("initializing...")
    solver, num_params, root, stamp = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    args = parse_option()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    project_name = "SQA"
    SQA_TRAIN = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_train.json")))
    SQA_VAL = json.load(open(os.path.join(CONF.PATH.SQA, project_name + "_val.json"))) 
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

    train(args, SQA_TRAIN, SQA_VAL, answer_counter_list)

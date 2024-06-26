"""
Modified from: https://github.com/daveredrum/ScanRefer/blob/master/lib/solver.py
"""

import os
import re
import sys
import time
import torch
import wandb
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn as nn

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from lib.eval_helper import get_eval
from utils.eta import decode_eta
from loss_helper import get_loss
from lib.pointnet2.pytorch_utils import BNMomentumScheduler
import pandas as pd
import transformers
import MinkowskiEngine as ME
from utils.visualization import visualize_scene_wandb

ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_vote_loss: {train_vote_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_box_loss: {train_box_loss}
[loss] train_sem_cls_loss: {train_sem_cls_loss}
[loss] train_aux_loss: {train_aux_loss}
[loss] train_answer_loss: {train_answer_loss}

[sco.] train_answer_acc@1: {train_answer_acc_at1}
[sco.] train_answer_acc@10: {train_answer_acc_at10}
[sco.] train_answer_acc_breakdown_what: {train_answer_acc_breakdown_what}
[sco.] train_answer_acc_breakdown_isare: {train_answer_acc_breakdown_isare}
[sco.] train_answer_acc_breakdown_how: {train_answer_acc_breakdown_how}
[sco.] train_answer_acc_breakdown_can: {train_answer_acc_breakdown_can}
[sco.] train_answer_acc_breakdown_which: {train_answer_acc_breakdown_which}
[sco.] train_answer_acc_breakdown_if: {train_answer_acc_breakdown_if}
[sco.] train_answer_acc_breakdown_where: {train_answer_acc_breakdown_where}
[sco.] train_answer_acc_breakdown_am: {train_answer_acc_breakdown_am}
[sco.] train_answer_acc_breakdown_other: {train_answer_acc_breakdown_other}
[sco.] train_situation_acc_0_5m: {train_situation_acc_0_5m}
[sco.] train_situation_acc_1_0m: {train_situation_acc_1_0m}
[sco.] train_situation_acc_15deg: {train_situation_acc_15deg}
[sco.] train_situation_acc_30deg: {train_situation_acc_30deg}

[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_vote_loss: {train_vote_loss}
[train] train_objectness_loss: {train_objectness_loss}
[train] train_box_loss: {train_box_loss}
[train] train_sem_cls_loss: {train_sem_cls_loss}
[train] train_aux_loss: {train_aux_loss}
[train] train_answer_loss: {train_answer_loss}

[train] train_answer_acc@1: {train_answer_acc_at1}
[train] train_answer_acc@10: {train_answer_acc_at10}
[train] train_answer_acc_breakdown_what: {train_answer_acc_breakdown_what}
[train] train_answer_acc_breakdown_isare: {train_answer_acc_breakdown_isare}
[train] train_answer_acc_breakdown_how: {train_answer_acc_breakdown_how}
[train] train_answer_acc_breakdown_can: {train_answer_acc_breakdown_can}
[train] train_answer_acc_breakdown_which: {train_answer_acc_breakdown_which}
[train] train_answer_acc_breakdown_if: {train_answer_acc_breakdown_if}
[train] train_answer_acc_breakdown_where: {train_answer_acc_breakdown_where}
[train] train_answer_acc_breakdown_am: {train_answer_acc_breakdown_am}
[train] train_answer_acc_breakdown_other: {train_answer_acc_breakdown_other}
[train] train_situation_acc_0_5m: {train_situation_acc_0_5m}
[train] train_situation_acc_1_0m: {train_situation_acc_1_0m}
[train] train_situation_acc_15deg: {train_situation_acc_15deg}
[train] train_situation_acc_30deg: {train_situation_acc_30deg}

[val]   val_loss: {val_loss}
[val]   val_vote_loss: {val_vote_loss}
[val]   val_objectness_loss: {val_objectness_loss}
[val]   val_box_loss: {val_box_loss}
[val]   val_sem_cls_loss: {train_sem_cls_loss}
[val]   val_aux_loss: {val_aux_loss}
[val]   val_answer_loss: {val_answer_loss}

[val]   val_answer_acc@1: {val_answer_acc_at1}
[val]   val_answer_acc@10: {val_answer_acc_at10}
[val]   val_answer_acc_breakdown_what: {val_answer_acc_breakdown_what}
[val]   val_answer_acc_breakdown_isare: {val_answer_acc_breakdown_isare}
[val]   val_answer_acc_breakdown_how: {val_answer_acc_breakdown_how}
[val]   val_answer_acc_breakdown_can: {val_answer_acc_breakdown_can}
[val]   val_answer_acc_breakdown_which: {val_answer_acc_breakdown_which}
[val]   val_answer_acc_breakdown_if: {val_answer_acc_breakdown_if}
[val]   val_answer_acc_breakdown_where: {val_answer_acc_breakdown_where}
[val]   val_answer_acc_breakdown_am: {val_answer_acc_breakdown_am}
[val]   val_answer_acc_breakdown_other: {val_answer_acc_breakdown_other}
[val]   val_situation_acc_0_5m: {val_situation_acc_0_5m}
[val]   val_situation_acc_1_0m: {val_situation_acc_1_0m}
[val]   val_situation_acc_15deg: {val_situation_acc_15deg}
[val]   val_situation_acc_30deg: {val_situation_acc_30deg}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] vote_loss: {vote_loss}
[loss] objectness_loss: {objectness_loss}
[loss] box_loss: {box_loss}
[loss] sem_cls_loss: {sem_cls_loss}
[loss] aux_loss: {aux_loss}
[loss] answer_loss: {answer_loss}

[sco.] answer_acc@1: {answer_acc_at1}
[sco.] answer_acc@10: {answer_acc_at10}
[sco.] answer_acc_breakdown_what: {answer_acc_breakdown_what}
[sco.] answer_acc_breakdown_isare: {answer_acc_breakdown_isare}
[sco.] answer_acc_breakdown_how: {answer_acc_breakdown_how}
[sco.] answer_acc_breakdown_can: {answer_acc_breakdown_can}
[sco.] answer_acc_breakdown_which: {answer_acc_breakdown_which}
[sco.] answer_acc_breakdown_if: {answer_acc_breakdown_if}
[sco.] answer_acc_breakdown_where: {answer_acc_breakdown_where}
[sco.] answer_acc_breakdown_am: {answer_acc_breakdown_am}
[sco.] answer_acc_breakdown_other: {answer_acc_breakdown_other}
[sco.] situation_acc_0_5m: {situation_acc_0_5m}
[sco.] situation_acc_1_0m: {situation_acc_1_0m}
[sco.] situation_acc_15deg: {situation_acc_15deg}
[sco.] situation_acc_30deg: {situation_acc_30deg}
"""

LOG_SCORE_KEYS = {
    "loss": ["loss", "vote_loss", "objectness_loss", "box_loss", "sem_cls_loss", "aux_loss", "answer_loss", "rot_loss", "pos_loss"],
    "score": ["answer_acc_at1", "answer_acc_at10", 
            "answer_acc_breakdown_what", "answer_acc_breakdown_isare", "answer_acc_breakdown_how", "answer_acc_breakdown_can", "answer_acc_breakdown_which", "answer_acc_breakdown_if", "answer_acc_breakdown_where", "answer_acc_breakdown_am", "answer_acc_breakdown_other",
            "situation_acc_0_5m", "situation_acc_1_0m", "situation_acc_15deg", "situation_acc_30deg",],
}

class Solver():
    def __init__(self, args, model, config, dataloader, optimizer, stamp, situation_loss_tag, val_step=10,
                cur_criterion="answer_acc_at1", detection=False, use_aux_situation=True, use_answer=True,
                max_grad_norm=None, lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None,
    ):
        self.args = args
        self.epoch = 0
        self.verbose = 0
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step
        self.cur_criterion = cur_criterion

        self.answerable_data_size = {}
        self.all_data_size = {}
        for phase in dataloader.keys():
            self.answerable_data_size[phase] = dataloader[phase].dataset.answerable_data_size
            self.all_data_size[phase] = dataloader[phase].dataset.all_data_size

        self.detection = detection
        self.use_answer = use_answer
        self.use_aux_situation = use_aux_situation

        self.max_grad_norm = max_grad_norm
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        self.situation_loss_tag = situation_loss_tag

        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "answer_loss": float("inf"),
            "aux_loss": float("inf"),
            "objectness_loss": float("inf"),
            "vote_loss": float("inf"),
            "box_loss": float("inf"),
            "sem_cls_loss": float("inf"),
            "answer_acc_at1": -float("inf"),
            "answer_acc_at10": -float("inf"),
            "answer_acc_breakdown_what": -float("inf"),
            "answer_acc_breakdown_isare": -float("inf"),
            "answer_acc_breakdown_how": -float("inf"),
            "answer_acc_breakdown_can": -float("inf"),
            "answer_acc_breakdown_which": -float("inf"),
            "answer_acc_breakdown_if": -float("inf"),
            "answer_acc_breakdown_where": -float("inf"),
            "answer_acc_breakdown_am": -float("inf"),
            "answer_acc_breakdown_other": -float("inf"),
            "situation_acc_0_5m": -float("inf"),
            "situation_acc_1_0m": -float("inf"),
            "situation_acc_15deg": -float("inf"),
            "situation_acc_30deg": -float("inf"),
        }

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }

        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if args.lr_scheduler_type == "step":
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            # NOTE: not implemented
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate**(int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)
        else:
            self.bn_scheduler = None

    def __call__(self, epoch, verbose):

        #######################################
        #                                     #
        #             START BLOCK             # 
        #                                     #
        #######################################
        self._start() 
        # setting
        self.epoch = epoch
        self.verbose = verbose

        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * self.val_step

        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))

                #######################################
                #                                     #
                #             FEED BLOCK              # 
                #                                     #
                #######################################
                self._feed(self.dataloader["train"], "train", epoch_id)

                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

                # update lr scheduler
                if self.lr_scheduler:
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        print("update learning rate for param_group[{}] --> {}\n".format(i, param_group['lr']))
                    self.lr_scheduler.step() 

                # update bn scheduler
                if self.bn_scheduler:
                    print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                    self.bn_scheduler.step()

            # If the training is interrupted by the user, finish training and exit
            except KeyboardInterrupt:
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _start(self):
        # save commandline
        cmd = " ".join([v for v in sys.argv])
        cmd_file = os.path.join(CONF.PATH.OUTPUT, self.stamp, "cmdline.txt")
        open(cmd_file, 'w').write(cmd)
        wandb.save(cmd_file)

    def _feed(self, dataloader, phase, epoch_id):
        # switch mode
        self._set_phase(phase)

        # re-init log
        self._reset_log(phase)

        scene_number_to_id = dataloader.dataset.scene_number_to_id

        # change dataloader
        dataloader = dataloader if phase == "train" else tqdm(dataloader)

        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                # import pdb; pdb.set_trace()
                if type(data_dict[key]) is dict or isinstance(data_dict[key], transformers.tokenization_utils_base.BatchEncoding):
                    # print("data_dict[key] is dict")
                    data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
                elif isinstance(data_dict[key], list):
                    data_dict[key] = data_dict[key]
                else:
                    # print("data_dict[key] is not dict")
                    data_dict[key] = data_dict[key].cuda()

            # initialize the running loss
            self._running_log = {
                # loss
                "loss": 0,
                "answer_loss": 0,
                "aux_loss": 0,
                "objectness_loss": 0,
                "vote_loss": 0,
                "box_loss": 0,
                "sem_cls_loss": 0,
                "pos_loss":0,
                "rot_loss":0,
                # score
                "answer_acc_at1": 0,
                "answer_acc_at10": 0,
                "answer_acc_breakdown_what": [0, 0],
                "answer_acc_breakdown_isare": [0, 0],
                "answer_acc_breakdown_how": [0, 0],
                "answer_acc_breakdown_can": [0, 0],
                "answer_acc_breakdown_which": [0, 0],
                "answer_acc_breakdown_if": [0, 0],
                "answer_acc_breakdown_where": [0, 0],
                "answer_acc_breakdown_am": [0, 0],
                "answer_acc_breakdown_other": [0, 0],
                # metric
                'situation_acc_0_5m': 0,
                'situation_acc_1_0m': 0,
                'situation_acc_15deg': 0,
                'situation_acc_30deg': 0,
            }

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

            with torch.autograd.set_detect_anomaly(True):
                # forward
                start = time.time()

                #######################################
                #                                     #
                #           FORWARD BLOCK             # 
                #                                     #
                #######################################
                data_dict = self._forward(data_dict)
                self._compute_loss(data_dict)
                self.log[phase]["forward"].append(time.time() - start)

                # backward
                if phase == "train":
                    start = time.time()
                    self._backward()
                    self.log[phase]["backward"].append(time.time() - start)

            # eval
            start = time.time()

            #######################################
            #                                     #
            #             EVAL BLOCK              # 
            #                                     #
            #######################################
            self._eval(data_dict, phase, scene_number_to_id)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            for key in self._running_log.keys():
                value = self._running_log[key] # score or loss
                if type(value) == torch.Tensor:
                    value = value.item() # if loss
                self.log[phase][key].append(value)  # breakdown scores have two values
            answerable_rate = self.answerable_data_size[phase] / self.all_data_size[phase]

            if "pred_answers" in data_dict:
                self.log[phase]["pred_answer"] += data_dict["pred_answers"].tolist()

            self.log[phase]["scene_id"] += [scene_number_to_id[scene_number] for scene_number in data_dict["scene_id"].tolist()]
            self.log[phase]["question_id"] += data_dict["question_id"].tolist()

            # log the time into wandb
            iter_time = self.log[phase]["fetch"][-1]
            iter_time += self.log[phase]["forward"][-1]
            if phase == "train":
                iter_time += self.log[phase]["backward"][-1]
            iter_time += self.log[phase]["eval"][-1]
            self.log[phase]["iter_time"].append(iter_time)

            time_group = ["fetch", "forward", "backward", "eval", "iter_time", "visualize_scene"]
            for key in time_group:
                value = np.mean([v for v in self.log[phase][key]]) if self.log[phase][key] != [] else 0
                wandb.log({"{}/time/{}".format(phase, key): value}, step=self._global_iter_id)

            # report
            if phase == "train":
                if (self._global_iter_id + 1) % self.verbose == 0:
                    self._train_report(epoch_id)

                # evaluation
                if self._global_iter_id % self.val_step == 0:
                    print("evaluating...")
                    # val
                    self._feed(self.dataloader["val"], "val", epoch_id)
                    self._dump_log("val")
                    self._set_phase("train")
                    self._epoch_report(epoch_id)   # Technically, this is not epoch report, but validation report

                # dump log
                self._dump_log("train")
                wandb.log({"params/epoch": epoch_id + 1,}, step=self._global_iter_id)
                if self.args.sep_params:
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        wandb.log({"params/sep_params_learning_rate_{}".format(i): param_group['lr'],}, step=self._global_iter_id)
                else:
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        wandb.log({"params/no_sep_learning_rate_{}".format(i): param_group['lr'],}, step=self._global_iter_id)
                self._global_iter_id += 1

        # check best
        if phase == "val":
            cur_best = np.mean(self.log[phase][self.cur_criterion])
            if cur_best > self.best[self.cur_criterion]:
                self._log("best val_{} achieved: {}".format(self.cur_criterion, np.abs(cur_best)))
                self._log("current train_loss: {}".format(np.mean(self.log["train"]["loss"])))
                self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
                self.best["epoch"] = epoch_id + 1

                for key in LOG_SCORE_KEYS["loss"] + LOG_SCORE_KEYS["score"]:
                    if 'breakdown' in key:
                        total = np.sum(self.log[phase][key], axis=0)
                        self.best[key] = total[0] / total[1] if total[1] != 0 else 0
                    else:
                        self.best[key] = np.mean(self.log[phase][key])

                # WandB logging of best_val_score
                for key, value in self.best.items():
                    wandb.log({"best_val/{}".format(key): round(value, 5)}, step=self._global_iter_id)

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)

                if "pred_answer" in self.log[phase]:
                    pred_answer_idxs = self.log[phase]["pred_answer"]
                    pred_answers = [self.dataloader["val"].dataset.answer_vocab.itos(pred_answer_idx) for pred_answer_idx in pred_answer_idxs]

                    qa_id_df = pd.DataFrame([self.log[phase]["scene_id"], self.log[phase]["question_id"]]).T
                    qa_id_df.columns = ["scene_id", "question_id"]
                    pred_ansewr_df = pd.DataFrame([pred_answer_idxs, pred_answers]).T
                    pred_ansewr_df.columns = ["pred_answer_idx", "pred_answer"]

                    # save pred_answers
                    pred_ansewr_df = pd.concat([qa_id_df, pred_ansewr_df], axis=1)
                    pred_ansewr_df.to_csv(os.path.join(model_root, "best_val_pred_answers.csv"), index=False)

                # save model
                torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))
    
    def _forward(self, data_dict):
        pcl = data_dict['point_clouds']
        feat = data_dict['point_colors']
        data_dict['openscene_in'] = ME.SparseTensor(feat.cuda(non_blocking=True), pcl.cuda(non_blocking=True))
        data_dict = self.model(data_dict)
        return data_dict

    def _compute_loss(self, data_dict):
        _, data_dict = get_loss(
            data_dict=data_dict,
            config=self.config,
            detection=self.detection,
            use_answer=self.use_answer,
            use_aux_situation=self.use_aux_situation,
            situation_loss_tag=self.situation_loss_tag,
        )
        # dump
        self._running_log["answer_loss"] = data_dict["answer_loss"]
        self._running_log["aux_loss"] = data_dict["aux_loss"]
        self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        self._running_log["vote_loss"] = data_dict["vote_loss"]
        self._running_log["box_loss"] = data_dict["box_loss"]
        self._running_log["sem_cls_loss"] = data_dict["sem_cls_loss"]
        self._running_log["loss"] = data_dict["loss"]
        self._running_log["pos_loss"] = data_dict["pos_loss"]
        self._running_log["rot_loss"] = data_dict["rot_loss"]

    def _eval(self, data_dict, phase, scene_number_to_id):
        data_dict = get_eval(
            data_dict=data_dict,
            config=self.config,
            answer_vocab=self.dataloader["train"].dataset.answer_vocab,
            use_aux_situation=self.use_aux_situation,
            situation_loss_tag=self.situation_loss_tag
        )

        # dump
        self._running_log["answer_acc_at1"] = data_dict["answer_acc_at1"].item()
        self._running_log["answer_acc_at10"] = data_dict["answer_acc_at10"].item()
        self._running_log["answer_acc_breakdown_what"] = data_dict["answer_acc_breakdown_what"].detach().tolist()
        self._running_log["answer_acc_breakdown_isare"] = data_dict["answer_acc_breakdown_isare"].detach().tolist()
        self._running_log["answer_acc_breakdown_how"] = data_dict["answer_acc_breakdown_how"].detach().tolist()
        self._running_log["answer_acc_breakdown_can"] = data_dict["answer_acc_breakdown_can"].detach().tolist()
        self._running_log["answer_acc_breakdown_which"] = data_dict["answer_acc_breakdown_which"].detach().tolist()
        self._running_log["answer_acc_breakdown_if"] = data_dict["answer_acc_breakdown_if"].detach().tolist()
        self._running_log["answer_acc_breakdown_where"] = data_dict["answer_acc_breakdown_where"].detach().tolist()
        self._running_log["answer_acc_breakdown_am"] = data_dict["answer_acc_breakdown_am"].detach().tolist()
        self._running_log["answer_acc_breakdown_other"] = data_dict["answer_acc_breakdown_other"].detach().tolist()

        self._running_log['situation_acc_0_5m'] = data_dict['situation_acc_0_5m'].item()
        self._running_log['situation_acc_1_0m'] = data_dict['situation_acc_1_0m'].item()
        self._running_log['situation_acc_15deg'] = data_dict['situation_acc_15deg'].item()
        self._running_log['situation_acc_30deg'] = data_dict['situation_acc_30deg'].item()

        # save visualizations to wandb
        start = time.time()
        self._visualize_scene_wandb(data_dict, scene_number_to_id, phase,)
        self.log[phase]["visualize_scene"].append(time.time() - start)

    def _visualize_scene_wandb(self, data_dict, scene_number_to_id, phase):
        # get scene
        scene_ids = [scene_number_to_id[scene_number] for scene_number in data_dict['scene_id'].tolist()]
        if (phase == "train" and CONF.VISUALIZATION.TRAIN) or (phase == "val" and CONF.VISUALIZATION.VAL):
            visualize_scene_wandb(data_dict, scene_ids, self.stamp, phase, self._global_iter_id, self.use_answer, self.situation_loss_tag)
        else:
            pass

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _reset_log(self, phase):
        self.log[phase] = {
            # info
            "forward": [],
            "backward": [],
            "eval": [],
            "fetch": [],
            "iter_time": [],
            "visualize_scene": [],
            # loss
            "loss": [],
            "answer_loss": [],
            "aux_loss": [],
            "objectness_loss": [],
            "vote_loss": [],
            "box_loss": [],
            "sem_cls_loss": [],
            "pos_loss": [],
            "rot_loss": [],
            # scores
            "answer_acc_at1": [],
            "answer_acc_at10": [],
            "answer_acc_breakdown_what": [],
            "answer_acc_breakdown_isare": [],
            "answer_acc_breakdown_how": [],
            "answer_acc_breakdown_can": [],
            "answer_acc_breakdown_which": [],
            "answer_acc_breakdown_if": [],
            "answer_acc_breakdown_where": [],
            "answer_acc_breakdown_am": [],
            "answer_acc_breakdown_other": [],
            # pred_answers
            "pred_answer": [],
            "scene_id": [],
            "question_id": [],
            # metrics
            'situation_acc_0_5m': [],
            'situation_acc_1_0m': [],
            'situation_acc_15deg': [],
            'situation_acc_30deg': [],
        }

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()

        # gradient clipping
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.max_grad_norm)

        self.optimizer.step()

    def _dump_log(self, phase):
        for loss_or_score in ["loss", "score"]:
            for key in LOG_SCORE_KEYS[loss_or_score]:
                if 'breakdown' in key:
                    total = np.sum(self.log[phase][key], axis=0)
                    value = total[0] / total[1]  if total[1] != 0 else 0
                else:
                    value = np.mean([v for v in self.log[phase][key]])
                # TensorBoard
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(loss_or_score, key),
                    value,
                    self._global_iter_id
                )
                # WandB
                # phase, key, item -> val/score/ref_acc
                wandb.log({"{}/{}/{}".format(phase, loss_or_score, key): value}, step=self._global_iter_id)


    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))
        wandb.save(os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))
        
        # finish the wandb model
        wandb.finish()
        

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
        eta_sec = (self._total_iter["train"] - self._global_iter_id - 1) * mean_train_time
        eta_sec += len(self.dataloader["val"]) * np.ceil(self._total_iter["train"] / self.val_step) * mean_est_val_time
        eta = decode_eta(eta_sec)

        iter_report_dic = {}
        phase = "train"
        for key in LOG_SCORE_KEYS["loss"] + LOG_SCORE_KEYS["score"]:
            if 'breakdown' in key:
                total = np.sum(self.log[phase][key], axis=0)
                iter_report_dic[phase+"_"+key] = round(total[0] / total[1], 5)  if total[1] != 0 else 0
            else:
                iter_report_dic[phase+"_"+key] = round(np.mean([v for v in self.log[phase][key]]), 5)
        iter_report_dic["epoch_id"] = epoch_id + 1
        iter_report_dic["iter_id"] = self._global_iter_id + 1
        iter_report_dic["total_iter"] = self._total_iter[phase]
        iter_report_dic["mean_fetch_time"] = round(np.mean(fetch_time), 5)
        iter_report_dic["mean_forward_time"] = round(np.mean(forward_time), 5)
        iter_report_dic["mean_backward_time"] = round(np.mean(backward_time), 5)
        iter_report_dic["mean_eval_time"] = round(np.mean(eval_time), 5)
        iter_report_dic["mean_iter_time"] = round(np.mean(iter_time), 5)
        iter_report_dic["eta_h"]=eta["h"]
        iter_report_dic["eta_m"]=eta["m"]
        iter_report_dic["eta_s"]=eta["s"]

        iter_report = self.__iter_report_template.format(**iter_report_dic)
        self._log(iter_report)


    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report_dic = {}
        for phase in ["train", "val"]:
            for key in LOG_SCORE_KEYS["loss"] + LOG_SCORE_KEYS["score"]:
                # NOTE: the regular expression is used to remove the leading 0 in the key, but seems useless here
                if 'breakdown' in key:
                    total = np.sum(self.log[phase][key], axis=0)
                    epoch_report_dic[phase + "_" + key] = round(total[0] / total[1], 5)  if total[1] != 0 else 0
                else:
                    epoch_report_dic[phase + "_" + key] = round(np.mean([v for v in self.log[phase][key]]), 5)
        epoch_report = self.__epoch_report_template.format(**epoch_report_dic)
        self._log(epoch_report)


    def _best_report(self):
        self._log("training completed...")
        best_report_dic = {k:v for k, v in self.best.items()}
        best_report = self.__best_report_template.format(**best_report_dic)
        # WandB logging of best_val_score
        for key, value in self.best.items():
            pass
            # wandb.log({"best_val/{}".format(key): round(value, 5)})

        self._log(best_report)
        best_report_file = os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt")
        with open(best_report_file, "w") as f:
            f.write(best_report)
        wandb.save(best_report_file)

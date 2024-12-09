#!/bin/bash

# IMPORTANT: This script should be run from the root directory of the project

# Change PROJECT_TAGS to the tags you want to use for the project

/usr/bin/python situation3d/train/train.py --tags PROJECT_TAGS --use_bert --finetune_bert_last_layer --gpu 0 --batch_size 32 --optim_name adamw --wd 0.05 --lr_scheduler_type step --lr_decay_step 15 20 25 --lr 2e-5 --situation_loss_tag __l2__quat__
#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    ## <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      ## <- or one of: cpu gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbsg-delta-gpu
#SBATCH --job-name=sig3d
#SBATCH --time=48:00:00      ## hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=yunzem2@illinois.edu
#SBATCH -o /scratch/bbsh/yunzem2/logs/sig3d/baseline-%j.out

. ~/.bashrc

module reset 
module list  # job documentation and metadata
module unload anaconda3_gpu
echo "job is starting on `hostname`"

conda activate /scratch/bbsg/yunzem2/conda-envs/sqa3d_dev
cd codes/3dvqa/SQA3D/ScanQA
WANDB_API_KEY=0ffc8e9f14980383f93bd2ba5e93425ef1ada469  WANDB_USERNAME=yunzeman python scripts/train.py --tags wd005_lrdecay15_lr2e5_situationL1Angle_openseg_mcan2  --use_bert --finetune_bert_last_layer  --gpu 0 --batch_size 16 --optim_name adamw --wd 0.02 --lr_scheduler_type step --lr_decay_step 15 20 25 --lr 2e-5 --situation_loss_tag  __l1__angle__  --mcan_num_layers 2
WANDB_API_KEY=0ffc8e9f14980383f93bd2ba5e93425ef1ada469  WANDB_USERNAME=yunzeman python scripts/train.py --tags wd005_lrdecay15_lr2e5_situationL2Quat_opensceneBottleneck_vtoken256_openseg --use_bert --finetune_bert_last_layer --gpu 0 --batch_size 32 --optim_name adamw --wd 0.05 --lr_scheduler_type step --lr_decay_step 15 20 25 --lr 2e-5 --situation_loss_tag __l2__quat__


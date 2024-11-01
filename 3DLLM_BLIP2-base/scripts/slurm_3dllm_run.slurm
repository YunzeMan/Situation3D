#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    ## <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      ## <- or one of: cpu gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbsg-delta-gpu
#SBATCH --job-name=3dllm_2dprojection
#SBATCH --time=48:00:00      ## hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=yunzem2@illinois.edu
#SBATCH -o /scratch/bbsh/yunzem2/logs/3dllm/lexicon3d/flant5-2dprojection-4gpu-v3.out

# flant5-2dprojection

. ~/.bashrc

module reset 
module list  # job documentation and metadata
module unload anaconda3_gpu
echo "job is starting on `hostname`"

conda activate /scratch/bbsg/yunzem2/conda-envs/3dllm
cd /u/yunzem2/codes/3dvqa/3D-LLM/3DLLM_BLIP2-base

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/pretrain.yaml

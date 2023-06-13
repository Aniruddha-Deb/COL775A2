#!/bin/bash

#PBS -N train_model
#PBS -P col775.cs1200869
#PBS -q test
#PBS -l select=1:ncpus=4:ngpus=2:mem=16G:centos=icelake
#PBS -l walltime=36:00:00
#PBS -l software=PYTORCH

# python runner for DDP
#PYTHON="torchrun --nproc_per_node=1 --master_port=$PORT"

echo "========================================"
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "========================================"

sleep 200000

# cd $PBS_O_WORKDIR
# module load apps/anaconda/3
# source activate dl
# module unload apps/anaconda/3
# 
# module load compiler/gcc/9.1.0
# module load compiler/gcc/9.1/openmpi/4.0.2
# module load compiler/cuda/11.0/compilervars
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.conda/envs/dl/lib
# export WANDB_MODE=offline
# 
# mpirun \
# -x MASTER_ADDR=$(head -n 1 $PBS_NODEFILE) \
# -x MASTER_PORT=$PORT \
# -bind-to none -map-by slot \
# -mca pml ob1 -mca btl ^openib \
# torchrun --nnodes=2 --nproc_per_node=1 \
# scripts/train.py --task base_slots --params slotformer/base_slots/configs/stosavi_clevrer_params.py --fp16 --ddp --cudnn
# 

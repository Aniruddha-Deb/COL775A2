#!/bin/bash

#PBS -N train_model
#PBS -P sit
#PBS -l select=2:ncpus=8:ngpus=1:mem=16G:centos=skylake
#PBS -l walltime=02:00:00
#PBS -l software=PYTORCH

echo "========================================"
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "========================================"

cd $PBS_O_WORKDIR
module load apps/anaconda/3
source activate dl
module unload apps/anaconda/3

module load compiler/gcc/9.1.0
module load compiler/gcc/9.1/openmpi/4.0.2
module load compiler/cuda/11.0/compilervars
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.conda/envs/dl/lib
export WANDB_MODE=offline

MASTER=`/bin/hostname -s`
cat $PBS_NODEFILE>nodelist
#Make sure this node (MASTER) comes first
SLAVES=`cat nodelist | grep -v $MASTER | uniq`

#We want names of master and slave nodes
HOSTLIST="$MASTER $SLAVES"

#The path you place your code
#This command to run your pytorch script
#You will want to replace this
#COMMAND="$path --world_size=$WORLD_SIZE"
COMMAND="scripts/train.py --task base_slots --params slotformer/base_slots/configs/stosavi_clevrer_params.py --fp16 --ddp --cudnn"

#Get a random unused port on this host(MASTER)
#First line gets list of unused ports
#3rd line gets single random port from the list
MPORT=`ss -tan | awk '{print $5}' | cut -d':' -f2 | \
        grep "[2-9][0-9]\{3,3\}" | sort | uniq | shuf -n 1`
NUMNODES=$(cat $PBS_NODEFILE | wc -l)

echo "Running on " $NUMNODES

#Launch the pytorch processes, first on master (first in $HOSTLIST) then on the slaves
RANK=0
echo $HOSTLIST
for node in $HOSTLIST; do
        ssh -q $node
                python3 -m torch.distributed.launch \
                --nproc_per_node=1 \
                --nnodes=$NUMNODES \
                --node_rank=$RANK \
                --master_addr="$MASTER" --master_port="$MPORT" \
                $COMMAND &
        RANK=$((RANK+1))
done
wait

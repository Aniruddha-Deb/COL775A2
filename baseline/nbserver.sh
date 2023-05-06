#!/bin/sh
### Set the job name (for your reference)
#PBS -N col775a2
### Set the project name, your department code by default
#PBS -P col775.cs1200869
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=4:mem=24G:ngpus=1
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=24:00:00

#PBS -l software=PYTORCH
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

HOME=/home/cse/btech/cs1200869

module load apps/anaconda/3
source activate dl_35
module unload apps/anaconda/3

module load compiler/gcc/9.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.conda/envs/dl_35/lib

cd $PBS_O_WORKDIR
jupyter lab --ip=e$(hostname).hpc.iitd.ac.in --no-browser

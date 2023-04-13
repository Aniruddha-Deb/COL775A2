#!/bin/sh
### Set the job name (for your reference)
#PBS -N train_bl
### Set the project name, your department code by default
#PBS -P col775.cs1200869
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=8:mem=24G:ngpus=1:centos=skylake
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=16:00:00

#PBS -l software=PYTORCH
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $PBS_WORKDIR
HOME=/home/cse/btech/cs1200869

module load apps/anaconda/3
source activate dl_35
module unload apps/anaconda/3

cd $HOME/proxy
python2 iitdlogin.py ani.cred&
PROXY_PID=$!

cd $HOME/scratch/COL775A2/baseline/code
module load compiler/gcc/9.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.conda/envs/dl_35/lib

python3 cnn-bert.py

kill $PROXY_PID

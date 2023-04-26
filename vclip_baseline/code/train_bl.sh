#!/bin/sh
### Set the job name (for your reference)
#PBS -N train_bl
### Set the project name, your department code by default
#PBS -P sit
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=16:mem=24G:ngpus=1:centos=skylake
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=24:00:00

#PBS -l software=PYTORCH
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $PBS_WORKDIR
HOME=/home/sit/phd/anz228400

module load apps/anaconda/3
source activate /scratch/sit/phd/anz228400/envs_conda/fariseq
module unload apps/anaconda/3

cd $HOME/proxy_login
python2 iitdlogin.py cred.txt&
PROXY_PID=$!

cd $HOME/scratch/col775/fairseq/examples/MMPT
module load compiler/gcc/9.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/sit/phd/anz228400/envs_conda/fariseq/lib/

python video-clip.py

kill $PROXY_PID

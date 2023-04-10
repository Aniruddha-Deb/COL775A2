#!/bin/sh
### Set the job name (for your reference)
#PBS -N frame_test
### Set the project name, your department code by default
#PBS -P col775.cs1200869
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=24:mem=16G:centos=skylake
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=00:45:00

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="

cd $PBS_WORKDIR
HOME=/home/cse/btech/cs1200869

#python2 $HOME/proxy/iitdlogin.py $HOME/proxy/ani.cred&
#proxy_pid=$!

#export http_proxy=http://10.10.78.22:3128
#export https_proxy=http://10.10.78.22:3128

cd $HOME/scratch/COL775A2/frames
module load apps/anaconda/3
source activate dl_35
module unload apps/anaconda/3

bash extract_test.sh

#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE

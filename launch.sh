#!/bin/bash
#SBATCH -A ees250003p
##SBATCH -p GPU
#SBATCH -p GPU-shared
#SBATCH -J inference
#SBATCH -N 1
##SBATCH --gpus=h100-80:1
#SBATCH --gpus=v100-32:4
#SBATCH --ntasks-per-node=4
##SBATCH --cpus-per-task=7
#SBATCH -t 00:20:00
#SBATCH -o inference-%j.out
#SBATCH -e inference-%j.out

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES


#ulimit -n 65536

module load anaconda3
conda activate orbit-e2s

export HOSTNAME=$(hostname)
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=7
time srun -n $((SLURM_JOB_NUM_NODES*1)) \
python ./launch_test.py


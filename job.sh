#!/bin/bash -l #
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00 
#SBATCH --job-name=mpa_2hr
#SBATCH --export=NONE 

unset SLURM_EXPORT_ENV 

# cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

#module load python/3.10-anaconda
source /home/hpc/iwal/iwal131h/spice/bin/activate

srun python train.py

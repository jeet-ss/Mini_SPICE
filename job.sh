#!/bin/bash -l #
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=01:00:00 
#SBATCH --job-name=mpa_mir1k
#SBATCH --export=NONE 

unset SLURM_EXPORT_ENV 

# cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load python/3.8-anaconda
source source /home/hpc/iwal/iwal131h/mpa/bin/activate

srun python python_setup_check.py

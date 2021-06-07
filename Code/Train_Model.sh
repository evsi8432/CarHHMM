#!/bin/bash
#SBATCH --account=def-nheckman

#SBATCH --mail-user=evan.sidrow@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

#SBATCH --time=72:00:00
#SBATCH --array=1-400
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1

if [ $SLURM_ARRAY_TASK_ID -lt 101 ]
then
  python Train_Model.py $SLURM_ARRAY_TASK_ID CarHMM
fi
if [ $SLURM_ARRAY_TASK_ID -gt 100 ] && [ $SLURM_ARRAY_TASK_ID -lt 201 ]
then
  python Train_Model.py $SLURM_ARRAY_TASK_ID HHMM
fi
if [ $SLURM_ARRAY_TASK_ID -gt 200 ] && [ $SLURM_ARRAY_TASK_ID -lt 301 ]
then
  python Train_Model.py $SLURM_ARRAY_TASK_ID CarHHMM1
fi
if [ $SLURM_ARRAY_TASK_ID -gt 300 ]
then
  python Train_Model.py $SLURM_ARRAY_TASK_ID CarHHMM2
fi

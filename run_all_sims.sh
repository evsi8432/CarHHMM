#!/bin/bash
#SBATCH --account=def-nheckman

#SBATCH --mail-user=evan.sidrow@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --array=1-100

python Run_Sim.py $SLURM_ARRAY_TASK_ID

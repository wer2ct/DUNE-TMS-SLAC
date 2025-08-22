#!/bin/bash
#SBATCH --job-name=hitclustering2500
#SBATCH --nodes=1
#SBATCH --account=neutrino:default
#SBATCH --partition=milano
#SBATCH --output=/sdf/data/neutrino/summer25/ktwall/logs/detsim_slurm-%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --gpus=0
#SBATCH --array=0-60
#SBATCH --qos=preemptable

apptainer exec \
  --env SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} \
  --env SLURM_JOB_ID=${SLURM_JOB_ID} \
  -B /sdf \
  /sdf/group/neutrino/images/develop.sif \
python3 python/TMS_HitClustering.py /sdf/data/neutrino/summer25/ktwall/full_spill_detsim/detsim_25${SLURM_ARRAY_TASK_ID}* /sdf/data/neutrino/summer25/ktwall/full_spill_clustered/ 25${SLURM_ARRAY_TASK_ID} config/TMSClusteringConfig.json

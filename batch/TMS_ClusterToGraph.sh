#!/bin/bash
#SBATCH --job-name=graphcreation2500
#SBATCH --nodes=1
#SBATCH --account=neutrino:default
#SBATCH --partition=milano
#SBATCH --output=/sdf/data/neutrino/summer25/ktwall/logs/tographs_slurm-%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15G
#SBATCH --time=00:45:00
#SBATCH --gpus=0
#SBATCH --array=0-60
#SBATCH --qos=preemptable

apptainer exec \
  --env SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} \
  --env SLURM_JOB_ID=${SLURM_JOB_ID} \
  -B /sdf \
  /sdf/group/neutrino/images/develop.sif \
python3 python/TMS_ClusterToGraph.py /sdf/data/neutrino/summer25/tanaka/nd-production/run-spill-build/MicroProdN4p1_NDComplex_FHC.spill.full/EDEPSIM_SPILLS/0002000/0002500/MicroProdN4p1_NDComplex_FHC.spill.full.00025${SLURM_ARRAY_TASK_ID}* /sdf/data/neutrino/summer25/ktwall/full_spill_clustered/hits_clustered_epsilon_0.1_25${SLURM_ARRAY_TASK_ID}* 25${SLURM_ARRAY_TASK_ID} /sdf/data/neutrino/summer25/ktwall/full_spill_graphs/

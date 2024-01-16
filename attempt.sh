#!/bin/bash
#SBATCH -J airways_urog_skin_TNFabundance__noGPU
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 00:00:00
#SBATCH -n 8
#SBATCH -p allgroups

cd /nfsd/bcb/bcbg/spina/
source tesi_env/bin/activate
cd CLAVAMB/vamb

srun 
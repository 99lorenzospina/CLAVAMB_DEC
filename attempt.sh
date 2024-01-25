#!/bin/bash
#SBATCH -J airways__pcabundance__estimatek
#SBATCH -o /nfsd/bcb/bcbg/spina/airways__pcabundance__nocontrastive__estimatek__noGPU/output_%j.txt
#SBATCH -e /nfsd/bcb/bcbg/spina/airways__pcabundance__nocontrastive__estimatek__noGPU/errors_%j.txt
#SBATCH -t 2:00:00
#SBATCH -n 5
#SBATCH -p allgroups
#SBATCH --mem 13G
#SBATCH -x runner-11

cd /nfsd/bcb/bcbg/spina/
source tesi_env/bin/activate
cd CLAVAMB/vamb

srun vamb --model aae --outdir /nfsd/bcb/bcbg/spina/airways__pcabundance__nocontrastive__estimatek__noGPU \
--fasta /nfsd/bcb/bcbg/spina/airways/contigs.fna.gz \
--rpkm /nfsd/bcb/bcbg/spina/airways/abundance.npz  \
--use_pc -o C
#!/bin/bash
#SBATCH -J airways_urog_skin__TNFabundance__nocontrastive__estimatek__noGPU__ONLYESTIM
#SBATCH -o output_%j.txt
#SBATCH -e errors_%j.txt
#SBATCH -t 13:00:00
#SBATCH -n 8
#SBATCH -p allgroups
#SBATCH -x runner-11

cd /nfsd/bcb/bcbg/spina/
source tesi_env/bin/activate
cd CLAVAMB/vamb

srun vamb --model aae --outdir /nfsd/bcb/bcbg/spina \
--fasta /nfsd/bcb/bcbg/spina/airways/contigs.fna.gz \
/nfsd/bcb/bcbg/spina/skin/contigs.fna.gz \
/nfsd/bcb/bcbg/spina/urog/contigs.fna.gz \
--rpkm /nfsd/bcb/bcbg/spina/airways/abundance.npz  \
/nfsd/bcb/bcbg/spina/skin/abundance.npz  \
/nfsd/bcb/bcbg/spina/urog/abundance.npz  \
-o C
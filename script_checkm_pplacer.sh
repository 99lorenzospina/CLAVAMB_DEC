#!/bin/bash
#SBATCH -J script_completo_checkm_pplacer
#SBATCH -o /nfsd/bcb/bcbg/spina/output_%j.txt
#SBATCH -e /nfsd/bcb/bcbg/spina/errors_%j.txt
#SBATCH -t 00:10:00
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH -x runner-11
#SBATCH --mem=250GB

handle_signals() {
    echo "Ricevuto segnale di interruzione, ma continua comunque con pplacer"
    # Sostituisci il percorso del tuo comando pplacer
    srun singularity exec -B /nfsd/bcb/bcbg/spina:/nfsd/bcb/bcbg/spina ./checkm_singularity.sif pplacer -j 1 -c \
    /nfsd/bcb/bcbg/spina/checkm_data_path/genome_tree/genome_tree_reduced.refpkg -o /nfsd/bcb/bcbg/spina/checkm_test_results/results/storage/tree/concatenated.pplacer.json \
    /nfsd/bcb/bcbg/spina/checkm_test_results/results/storage/tree/concatenated.fasta

    srun singularity exec -B /nfsd/bcb/bcbg/spina:/nfsd/bcb/bcbg/spina ./checkm_singularity.sif guppy tog -o \
    /nfsd/bcb/bcbg/spina/checkm_test_results/results/storage/tree/concatenated.tre \
    /nfsd/bcb/bcbg/spina/checkm_test_results/results/storage/tree/concatenated.pplacer.json

    srun singularity exec /nfsd/bcb/bcbg/spina/checkm_singularity.sif checkm test /nfsd/bcb/bcbg/spina/checkm_test_results
}
# Trap dei segnali
trap 'handle_signals' SIGINT SIGTERM

cd /nfsd/bcb/bcbg/spina/
source tesi_env/bin/activate

srun singularity exec /nfsd/bcb/bcbg/spina/checkm_singularity.sif checkm test /nfsd/bcb/bcbg/spina/checkm_test_results || true

srun singularity exec -B /nfsd/bcb/bcbg/spina:/nfsd/bcb/bcbg/spina ./checkm_singularity.sif pplacer -j 1 -c \
/nfsd/bcb/bcbg/spina/checkm_data_path/genome_tree/genome_tree_reduced.refpkg -o /nfsd/bcb/bcbg/spina/checkm_test_results/results/storage/tree/concatenated.pplacer.json \
/nfsd/bcb/bcbg/spina/checkm_test_results/results/storage/tree/concatenated.fasta

srun singularity exec -B /nfsd/bcb/bcbg/spina:/nfsd/bcb/bcbg/spina ./checkm_singularity.sif guppy tog -o \
/nfsd/bcb/bcbg/spina/checkm_test_results/results/storage/tree/concatenated.tre \
/nfsd/bcb/bcbg/spina/checkm_test_results/results/storage/tree/concatenated.pplacer.json

srun singularity exec /nfsd/bcb/bcbg/spina/checkm_singularity.sif checkm test /nfsd/bcb/bcbg/spina/checkm_test_results
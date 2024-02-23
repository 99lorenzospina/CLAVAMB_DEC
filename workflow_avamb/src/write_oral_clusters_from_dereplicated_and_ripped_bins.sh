#!/usr/bin/bash
#SBATCH -J ck_oral_transfer
#SBATCH -o /nfsd/bcb/bcbg/spina/output_%j.txt
#SBATCH -e /nfsd/bcb/bcbg/spina/errors_%j.txt
#SBATCH -t 00:05:00
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH --mem 10G
#SBATCH -x runner-11

# Definizione dei percorsi dei bin finali e del file di output dei cluster
drep_dir="/nfsd/bcb/bcbg/spina/avamb_checkm_results/oral/Final_bins/"

output_file="/nfsd/bcb/bcbg/spina/avamb_checkm_results/oral/avamb_manual_drep_disjoint_clusters.tsv"
echo 'creating z y v clusters from the final set of bins'
for s in $(ls $drep_dir)
do
s="$drep_dir"/"$s"/
if [ -d "$s" ]
then
cd $s
for bin in $(ls . 2> /dev/null)

do
if [[ $bin == **".fna" ]]
then

cluster_name=$(echo $bin | sed 's=.fna==g' | sed 's=.fa==g')

echo -e   "clustername\tcontigname"  >> $output_file
for contig in $(grep '>' $bin | sed 's=>==g')
do
echo -e   "$cluster_name""\t""$contig"  >> $output_file
done


fi
done

fi
done
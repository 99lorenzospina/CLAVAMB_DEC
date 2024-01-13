import sys
import os
import numpy as np
import torch
import random
from pyclustering.cluster.gmeans import gmeans
from Bio import SeqIO
import gzip

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
import vamb

with vamb.vambtools.Reader(os.path.join(parentdir, 'test', 'data', 'contigs.fna.gz')) as file:
        composition = vamb.parsecontigs.Composition.from_file(file, minlength=100, use_pc= True)
        file.close()

tnf = composition.matrix
rpkm = np.ones_like(tnf[:, :3])
lengths = composition.metadata.lengths
dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, lengths, batchsize=16)

estimator = gmeans(np.concatenate((tnf, rpkm), axis=1), ccore = True)
estimator.process()
nlatent_aae_y = len(estimator.get_clusters())
print(nlatent_aae_y)


def guess_clusters_num(sequences, init_K=4, end_situation=0.1):
    """
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as score
    
    last_score = abs(score(sequences, KMeans(init_K).fit_predict(sequences)) -
                     end_situation)
    while init_K < len(sequences) // 250:
        init_K += 1
        new_score = abs(score(sequences,
                              KMeans(init_K).fit_predict(sequences)) -
                        end_situation)
        if new_score > last_score:
            break
        last_score = new_score
    return init_K

sequences = []
with gzip.open(os.path.join(parentdir, 'test', 'data', 'contigs.fna.gz'), "rt") as handle:
        for sequence in SeqIO.parse(handle, "fasta"):
            # Itera attraverso le sequenze e stampa le informazioni
            print(f"Header: {sequence.id}")
            print(f"Sequence: {sequence.seq}")
            sequences.append(sequence)
            print(f"Length: {len(sequence)}")
            # Aggiungi altre informazioni se necessario
            print("\n")

print(guess_clusters_num(sequences))

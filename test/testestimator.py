import sys
import os
import numpy as np
import torch
import random
from pyclustering.cluster.gmeans import gmeans
from Bio import SeqIO
import gzip
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as score
import psutil

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
import vamb
'''
'''
'''
def guess_clusters_num(sequences, init_K=4, end_situation=0.1):
    
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
'''
logpath = os.path.join("/home/lorenzo/CLAVAMB_DEC/test/data", "log.txt")
tnf = np.ones((10, 9))
rpkm = np.ones((10, 8))
with open(logpath, "w") as logfile:
    estimator = vamb.species_number.gmeans(np.concatenate((tnf, rpkm), axis=1), logfile, ccore = False)
    estimator.process()
    nlatent_aae_y = len(estimator.get_definitive_centers())
    print(nlatent_aae_y)
with vamb.vambtools.Reader(os.path.join(parentdir, 'test', 'data', 'contigs.fna.gz')) as file:
        composition = vamb.parsecontigs.Composition.from_file(file, minlength=100, use_pc= False)
        file.close()

tnf = composition.matrix
rpkm = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'abundance.npz'))
lengths = composition.metadata.lengths
dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, lengths, batchsize=16)
with open(logpath, "w") as logfile:
    print(f"\nEstimate the number of clusters")
    begintime = time.time()/60
    estimator = vamb.species_number.gmeans(np.concatenate((tnf, rpkm), axis=1), logfile, ccore = False)
    estimator.process()
    nlatent_aae_y = len(estimator.get_definitive_centers())
    print(nlatent_aae_y)    #3502 (airways, tnf+abundance), 2267 (airways, pc+abundance)
    timepoint_gernerate_input=time.time()/60
    time_generating_input= round(timepoint_gernerate_input-begintime,2)
    print(f"\nCluster estimated in {time_generating_input}")

print(f"Numero di thread: {psutil.cpu_count(logical=False)}")
print(f"Uso della CPU: {psutil.cpu_percent()}%")
print(f"Uso della memoria: {psutil.virtual_memory().percent}%")
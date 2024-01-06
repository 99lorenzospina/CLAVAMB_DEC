import sys
import os
import numpy as np
import random

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
import vamb

outdir = './data/'
# Test making the dataloader
tnf = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_tnf.npz'))
rpkm = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_rpkm.npz'))
lengths = np.ones(tnf.shape[0])
lengths = np.exp((lengths + 5.0).astype(np.float32))
dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, lengths, batchsize=64)

# Can instantiate the VAE
vae = vamb.encode.VAE(103, nsamples=3)

# Can instantiate the AAE
aae = vamb.aamb_encode.AAE(103, nsamples=3)

# Training model works in general
tnf = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_tnf.npz'))
rpkm = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_rpkm.npz'))
lengths = np.ones(tnf.shape[0])
lengths = np.exp((lengths + 5.0).astype(np.float32))
dataloader, mask = vamb.encode.make_dataloader(rpkm, tnf, lengths, batchsize=16)


vae.trainmodel(dataloader, batchsteps=[5, 10], nepochs=15)
aae.trainmodel(dataloader, batchsteps=[5, 10], nepochs=15)

contignames = []
while len(contignames) < tnf.shape[0]:
    nuovo_codice = str(random.randint(100, 999))
    if nuovo_codice not in contignames:
        contignames.append(nuovo_codice)

latent = vae.encode(dataloader)
vamb.vambtools.write_npz(os.path.join(outdir, "target_vae_latent.npz"), latent)
cluster_y, latent_aae = aae.get_latents(contignames, dataloader)
vamb.vambtools.write_npz(os.path.join(outdir, "target_aae_z_latent.npz"), latent_aae)


cluster_generator = vamb.cluster.ClusterGenerator(
        latent,
        windowsize=200,
        minsuccesses=20,
        destroy=True,
        normalized=False,
        cuda=False,
    )

renamed = (
    (str(cluster_index + 1), {contignames[i] for i in members})
    for (cluster_index, (_, members)) in enumerate(
        map(lambda x: x.as_tuple(), cluster_generator)
    )
)

separator = None

# Binsplit if given a separator
if separator is not None:
    maybe_split = vamb.vambtools.binsplit(renamed, separator)
else:
    maybe_split = renamed

clusterspath = os.path.join(outdir, "target_vae_clusters.tsv")

with open(clusterspath, "w") as clustersfile:
    clusternumber, ncontigs = vamb.vambtools.write_clusters(
        clustersfile,
        maybe_split,
        max_clusters=None,
        min_size=1,
        rename=False,
        cluster_prefix='vae_'
    )

cluster_generator = vamb.cluster.ClusterGenerator(
        latent_aae,
        windowsize=200,
        minsuccesses=20,
        destroy=True,
        normalized=False,
        cuda=False,
    )

clusterspath = os.path.join(outdir, "target_aae_clusters.tsv")

maybe_split = (
        (str(cluster_index + 1), {contignames[i] for i in members})
        for (cluster_index, (_, members)) in enumerate(
            map(lambda x: x.as_tuple(), cluster_generator)
        )
    )

with open(clusterspath, "w") as clustersfile:
    clusternumber, ncontigs = vamb.vambtools.write_clusters(
        clustersfile,
        maybe_split,
        max_clusters=None,
        min_size=1,
        rename=False,
        cluster_prefix='aae_z_'
    )

clusterspath= os.path.join(outdir, "target_aae_y_clusters.tsv") 

maybe_split = cluster_y

with open(clusterspath, "w") as clustersfile:
            clusternumber, ncontigs = vamb.vambtools.write_clusters(
                clustersfile,
                maybe_split,
                max_clusters=None,
                min_size=1,
                rename=False,
                cluster_prefix='aae_y_'
            )
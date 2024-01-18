#!/usr/bin/env python3
'''
vamb --model aae --outdir /home/lorenzo/ \
--fasta /home/lorenzo/airways/contigs.fna.gz /home/lorenzo/skin/contigs.fna.gz /home/lorenzo/urog/contigs.fna.gz \
--rpkm /home/lorenzo/airways/abundance.npz /home/lorenzo/skin/abundance.npz /home/lorenzo/urog/abundance.npz  \
-o C
'''

# More imports below, but the user's choice of processors must be parsed before
# numpy can be imported.
import vamb
import numpy as np
import sys
import os
import argparse
import torch
import datetime
import time
import shutil
import math
from math import isfinite
from argparse import Namespace
from typing import Optional, IO
import warnings
from glob import glob
import random

_ncpu = os.cpu_count()
DEFAULT_THREADS = 8 if _ncpu is None else min(_ncpu, 8)

# These MUST be set before importing numpy
# I know this is a shitty hack, see https://github.com/numpy/numpy/issues/11826
os.environ["MKL_NUM_THREADS"] = str(DEFAULT_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(DEFAULT_THREADS)
os.environ["OMP_NUM_THREADS"] = str(DEFAULT_THREADS)

# Append vamb to sys.path to allow vamb import even if vamb was not installed
# using pip
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)


from vamb import aamb_encode

################################# DEFINE FUNCTIONS ##########################


def log(string: str, logfile: IO[str], indent: int = 0):
    print(("\t" * indent) + string, file=logfile)
    logfile.flush()


def calc_tnf(
    outdir: str,
    fastapath: Optional[str],
    npzpath: Optional[str],
    mincontiglength: int,
    logfile: IO[str],
    nepochs: int,
    augmentation_store_dir: str,
    augmode = [-1,-1],
    contrastive=True,
    k=4,
    use_pc=False
) -> vamb.parsecontigs.Composition:
    begintime = time.time()/60
    log("\nLoading TNF/PC", logfile, 0)
    log(f"Minimum sequence length: {mincontiglength}", logfile, 1)

    if npzpath is not None:
        log(f"Loading composition from npz {npzpath}", logfile, 1)
        composition = vamb.parsecontigs.Composition.load(npzpath)
        composition.filter_min_length(mincontiglength)
    else:
        assert fastapath is not None
        if contrastive:
            index_list_one = list(range(backup_iteration))
            random.shuffle(index_list_one)
            index_list_two = list(range(backup_iteration))
            random.shuffle(index_list_two)
            index_list = [index_list_one, index_list_two]
        if isinstance(fastapath, str):
            log(f"Loading data from FASTA file {fastapath}", logfile, 1)
            if not contrastive:
                with vamb.vambtools.Reader(fastapath) as file:
                    composition = vamb.parsecontigs.Composition.from_file(
                        file, minlength=mincontiglength, use_pc = use_pc
                    )
                    file.close()
            else:
                os.system(f'mkdir -p {augmentation_store_dir}')
                backup_iteration = math.ceil(math.sqrt(nepochs))
                log('Generating {} augmentation data'.format(backup_iteration), logfile, 1)
                with vamb.vambtools.Reader(fastapath) as file:
                    composition = vamb.parsecontigs.Composition.read_contigs_augmentation(
                        file, minlength=mincontiglength, k=k, index_list = index_list, store_dir=augmentation_store_dir, backup_iteration=backup_iteration, augmode=augmode, use_pc = use_pc)
                    file.close()
            composition.save(os.path.join(outdir, "composition.npz"))
        #multiple files in input (multiple datasets)
        else:
            log(f"Loading data from FASTA files {fastapath}", logfile, 1)
            b = True
            for path in fastapath:
                if not contrastive:
                    with vamb.vambtools.Reader(path) as file:
                        if b:
                            b = False
                            composition = None
                        composition = vamb.parsecontigs.Composition.concatenate(composition, vamb.parsecontigs.Composition.from_file(
                            file, minlength=mincontiglength, use_pc=use_pc
                        ))
                        file.close()    
                if contrastive:
                    os.system(f'mkdir -p {augmentation_store_dir}')
                    backup_iteration = math.ceil(math.sqrt(nepochs))
                    log('Generating {} augmentation data'.format(backup_iteration), logfile, 1)
                    with vamb.vambtools.Reader(fastapath) as file:
                        #Generate the composition for this path and update the overall augmentation files
                        newcomp = vamb.parsecontigs.Composition.read_contigs_augmentation(
                            file, minlength=mincontiglength, k=k, index_list = index_list, store_dir=augmentation_store_dir, backup_iteration=backup_iteration, augmode=augmode, use_pc = use_pc, already = not b)
                        if b:
                            b = False
                            composition = None
                        #Update the overall composition
                        composition = vamb.parsecontigs.Composition.concatenate(composition, newcomp
                        )
                        file.close()
            composition.save(os.path.join(outdir, "composition.npz"))

    ''' composition.save should do the trick
    vamb.vambtools.write_npz(os.path.join(outdir, 'tnf.npz'), tnfs)
    vamb.vambtools.write_npz(os.path.join(outdir, 'lengths.npz'), contiglengths)
    with open(os.path.join(outdir, 'contignames.txt'),'w') as f:
            f.write('\n'.join(contignames))
            f.close()
    '''
    
    elapsed = round(time.time()/60 - begintime, 2)
    print("", file=logfile)
    log(
        f"Kept {composition.count_bases()} bases in {composition.nseqs} sequences",
        logfile,
        1,
    )
    log(f"Processed TNF in {elapsed} minutes", logfile, 1)

    return composition


def calc_rpkm(
    outdir, bampaths, rpkmpath, jgipath, refhash, ncontigs, mincontiglength,
              minalignscore, minid, subprocesses, logfile
):
    begintime = time.time()
    log('\nLoading RPKM', logfile)
    # If rpkm is given, we load directly from .npz file
    if rpkmpath is not None:
        if isinstance(rpkmpath, str):
            log('Loading RPKM from npz array {}'.format(rpkmpath), logfile, 1)
            rpkms = vamb.vambtools.read_npz(rpkmpath)

            if not rpkms.dtype == np.float32:
                raise ValueError('RPKMs .npz array must be of float32 dtype')
        #multiple files in input (multiple datasets)
        else:
            print("Loading data from FASTA files {}".format(rpkmpath), file=logfile)
            old = np.array([])
            for path in rpkmpath:
                log('Loading RPKM from npz array {}'.format(path), logfile, 1)
                rpkms = vamb.vambtools.read_npz(path)
                if len(old) != 0:
                    rpkms = vamb.parsebam.avg_window(rpkms)
                    rpkms = np.concatenate((old, rpkms))
                old = rpkms
                if not rpkms.dtype == np.float32:
                    raise ValueError('RPKMs .npz array must be of float32 dtype')
            del old
    
    else:
        log('Reference hash: {}'.format(refhash if refhash is None else refhash.hex()), logfile, 1)
        # Else if JGI is given, we load from that
        if jgipath is not None:
            log('Loading RPKM from JGI file {}'.format(jgipath), logfile, 1)
            with open(jgipath) as file:
                rpkms = vamb.vambtools._load_jgi(file, mincontiglength, refhash)

        else:
            log('Parsing {} BAM files with {} subprocesses'.format(len(bampaths) if bampaths is not None else 0, subprocesses),
            logfile, 1)
            log('Min alignment score: {}'.format(minalignscore), logfile, 1)
            log('Min identity: {}'.format(minid), logfile, 1)
            log('Min contig length: {}'.format(mincontiglength), logfile, 1)
            log('\nOrder of columns is:', logfile, 1)
            log('\n\t'.join(bampaths), logfile, 1)
            print('', file=logfile)

            dumpdirectory = os.path.join(outdir, 'tmp')
            rpkms = vamb.parsebam.read_bamfiles(bampaths, dumpdirectory=dumpdirectory,
                                                refhash=refhash, minscore=minalignscore,
                                                minlength=mincontiglength, minid=minid,
                                                subprocesses=subprocesses, logfile=logfile)
            print('', file=logfile)
            vamb.vambtools.write_npz(os.path.join(outdir, 'rpkm.npz'), rpkms)
            shutil.rmtree(dumpdirectory)

    if len(rpkms) != ncontigs:
        raise ValueError("Length of TNFs and length of RPKM does not match. Verify the inputs")

    elapsed = round(time.time() - begintime, 2)
    log('Processed RPKM in {} seconds'.format(elapsed), logfile, 1)

    return rpkms

def trainvae(
    outdir: str,
    rpkms: np.ndarray,
    tnfs: np.ndarray,
    k: int,
    contrastive: bool,
    augmode: list[int],
    augdatashuffle: bool,
    augmentationpath: str,
    temperature: float,
    lengths: np.ndarray,
    nhiddens: Optional[list[int]],  # set automatically if None
    nlatent: int,
    alpha: Optional[float],  # set automatically if None
    beta: float,
    dropout: Optional[float],  # set automatically if None
    cuda: bool,
    batchsize: int,
    nepochs: int,
    lrate: float,
    batchsteps: list[int],
    logfile: IO[str],
) -> tuple[np.ndarray, np.ndarray]:

    begintime = time.time()/60
    log("\nCreating and training VAE", logfile)

    assert len(rpkms) == len(tnfs)

    nsamples = rpkms.shape[1]

    # basic config for contrastive learning
    aug_all_method = ['GaussianNoise','Transition','Transversion','Mutation','AllAugmentation']
    hparams = Namespace(
        validation_size=4096,   # Debug only. Validation size for training.
        visualize_size=25600,   # Debug only. Visualization (pca) size for training.
        temperature=temperature,        # The parameter for contrastive loss
        augmode=augmode,        # Augmentation method choices (in aug_all_method)
        sigma = 4000,           # Add weight on the contrastive loss to avoid gradient disappearance
        lrate_decent = 0.8,     # Decrease the learning rate by lrate_decent for each batchstep
        augdatashuffle = augdatashuffle     # Shuffle the augmented data for training to introduce more noise. Setting True is not recommended. [False]
    )

    dataloader, mask = vamb.encode.make_dataloader(
        rpkms, tnfs, lengths, batchsize, destroy=True, cuda=cuda
    )
    log("Created dataloader and mask", logfile, 1)
    vamb.vambtools.write_npz(os.path.join(outdir, "mask.npz"), mask)
    n_discarded = len(mask) - mask.sum()
    log(f"Number of sequences unsuitable for encoding: {n_discarded}", logfile, 1)
    log(f"Number of sequences remaining: {len(mask) - n_discarded}", logfile, 1)
    print("", file=logfile)

    if contrastive:
        if True:
            vae = vamb.encode.VAE(ntnf=int(tnfs.shape[1]), nsamples=nsamples, k=k, nhiddens=nhiddens, nlatent=nlatent,alpha=alpha, beta=beta, dropout=dropout, cuda=cuda, c=True)
            log("Created VAE", logfile, 1)
            modelpath = os.path.join(outdir, f"{aug_all_method[hparams.augmode[0]]+'_'+aug_all_method[hparams.augmode[1]]}_vae.pt")
            vae.trainmodel(dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps, logfile=logfile, modelfile=modelpath, hparams=hparams, augmentationpath=augmentationpath, mask=mask)
        else:
            modelpath = os.path.join(outdir, f"final-dim/{aug_all_method[hparams.augmode[0]]+' '+aug_all_method[hparams.augmode[1]]+' '+str(hparams.hidden_mlp)}_vae.pt")
            vae = vamb.encode.VAE.load(modelpath,cuda=cuda,c=True)
            log("Loaded VAE", logfile, 1)
            vae.to(('cuda' if cuda else 'cpu'))
    else:
        vae = vamb.encode.VAE(ntnf=int(tnfs.shape[1]), nsamples=nsamples, k=k, nhiddens=nhiddens, nlatent=nlatent, alpha=alpha, beta=beta, dropout=dropout, cuda=cuda)
        log("Created VAE", logfile, 1)
        modelpath = os.path.join(outdir, 'vae_model.pt')
        vae.trainmodel(dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps, logfile=logfile, modelfile=modelpath)

    print("", file=logfile)
    log("Encoding to latent representation", logfile, 1)
    latent = vae.encode(dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, "latent.npz"), latent)
    del vae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time()/60 - begintime, 2)
    log(f"Trained VAE and encoded in {elapsed} minutes", logfile, 1)

    return mask, latent

def trainaae(
    outdir: str,
    rpkms: np.ndarray,
    tnfs: np.ndarray,
    k: int,
    contrastive: bool,
    augmode: list[int],
    augdatashuffle: bool,
    augmentationpath: str,
    temperature: float,
    lengths: np.ndarray,
    nhiddens: Optional[list[int]],  # set automatically if None
    nlatent_z: int,
    nlatent_y: int,
    alpha: Optional[float],  # set automatically if None
    sl: float,
    slr: float,
    cuda: bool,
    batchsize: int,
    nepochs: int,
    lrate: float,
    batchsteps: list[int],
    logfile: IO[str],
    contignames: np.ndarray
) -> tuple[np.ndarray, np.ndarray,dict()]:

    begintime = time.time()/60
    log("\nCreating and training AAE", logfile)
    nsamples = rpkms.shape[1]    #number of contigs

    # basic config for contrastive learning
    aug_all_method = ['GaussianNoise','Transition','Transversion','Mutation','AllAugmentation']
    hparams = Namespace(
        validation_size=4096,   # Debug only. Validation size for training.
        visualize_size=25600,   # Debug only. Visualization (pca) size for training.
        temperature=temperature,        # The parameter for contrastive loss
        augmode=augmode,        # Augmentation method choices (in aug_all_method)
        sigma = 4000,           # Add weight on the contrastive loss to avoid gradient disappearance
        lrate_decent = 0.8,     # Decrease the learning rate by lrate_decent for each batchstep
        augdatashuffle = augdatashuffle     # Shuffle the augmented data for training to introduce more noise. Setting True is not recommended. [False]
    )
    
    assert len(rpkms) == len(tnfs)

    dataloader, mask = vamb.encode.make_dataloader(
        rpkms, tnfs, lengths, batchsize, destroy=True, cuda=cuda
    )
    log("Created dataloader and mask", logfile, 1)
    #vamb.vambtools.write_npz(os.path.join(outdir, "mask.npz"), mask)
    n_discarded = len(mask) - mask.sum()
    log(f"Number of sequences unsuitable for encoding: {n_discarded}", logfile, 1)
    log(f"Number of sequences remaining: {len(mask) - n_discarded}", logfile, 1)
    print("", file=logfile)

    if contrastive:
        if True:
            aae = vamb.aamb_encode.AAE(ntnf=int(tnfs.shape[1]), nsamples=nsamples, nhiddens=nhiddens, nlatent_l=nlatent_z, nlatent_y=nlatent_y, alpha=alpha, sl=sl, slr=slr, cuda=cuda, k=k, contrast=True)
            log("Created AAE", logfile, 1)
            modelpath = os.path.join(outdir, f"{aug_all_method[hparams.augmode[0]]+'_'+aug_all_method[hparams.augmode[1]]}_aae.pt")
            aae.trainmodel(dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps, logfile=logfile, modelfile=modelpath, hparams=hparams, augmentationpath=augmentationpath, mask=mask)
        else:
            modelpath = os.path.join(outdir, f"final-dim/{aug_all_method[hparams.augmode[0]]+' '+aug_all_method[hparams.augmode[1]]+' '+str(hparams.hidden_mlp)}_aae.pt")
            log("Loaded AAE", logfile, 1)
            aae = vamb.aamb_encode.AAE.load(modelpath,cuda=cuda,c=True)
            aae.to(('cuda' if cuda else 'cpu'))
    else:
        aae = vamb.aamb_encode.AAE(ntnf=int(tnfs.shape[1]), nsamples=nsamples, k=k, nhiddens=nhiddens, nlatent_l=nlatent_z, nlatent_y=nlatent_y, alpha=alpha, sl=sl, slr=slr, cuda=cuda, contrast=False)
        log("Created AAE", logfile, 1)
        modelpath = os.path.join(outdir, 'aae_model.pt')
        aae.trainmodel(dataloader, nepochs=nepochs, lrate=lrate, batchsteps=batchsteps, logfile=logfile, modelfile=modelpath)
    
    print("", file=logfile)
    log("Encoding to latent representation", logfile, 1)
    clusters_y_dict,latent = aae.get_latents(contignames, dataloader)
    vamb.vambtools.write_npz(os.path.join(outdir, "aae_z_latent.npz"), latent)
    #vamb.vambtools.write_npz(os.path.join(outdir, "aae_y_latent.npz"), clusters_y_dict) #this is computed at the end of cluster()

    del aae  # Needed to free "latent" array's memory references?

    elapsed = round(time.time()/60 - begintime, 2)
    log(f"Trained AAE and encoded in {elapsed} minutes", logfile, 1)

    return mask, latent, clusters_y_dict


def cluster(
    clusterspath: str,
    latent: np.ndarray,
    contignames: np.ndarray,  # of dtype object
    windowsize: int,
    minsuccesses: int,
    maxclusters: Optional[int],
    minclustersize: int,
    separator: Optional[str],
    cuda: bool,
    logfile: IO[str],
    cluster_prefix : str
) -> None:
    begintime = time.time()/60

    log("\nClustering", logfile)
    log(f"Windowsize: {windowsize}", logfile, 1)
    log(f"Min successful thresholds detected: {minsuccesses}", logfile, 1)
    log(f"Max clusters: {maxclusters}", logfile, 1)
    log(f"Min cluster size: {minclustersize}", logfile, 1)
    log(f"Use CUDA for clustering: {cuda}", logfile, 1)
    log(
        "Separator: {}".format(None if separator is None else ('"' + separator + '"')),
        logfile,
        1,
    )

    cluster_generator = vamb.cluster.ClusterGenerator(
        latent,
        windowsize=windowsize,
        minsuccesses=minsuccesses,
        destroy=True,
        normalized=False,
        cuda=cuda,
    )

    renamed = (
        (str(cluster_index + 1), {contignames[i] for i in members})
        for (cluster_index, (_, members)) in enumerate(
            map(lambda x: x.as_tuple(), cluster_generator)
        )
    )

    # Binsplit if given a separator
    if separator is not None:
        maybe_split = vamb.vambtools.binsplit(renamed, separator)
    else:
        maybe_split = renamed
    
    with open(clusterspath, "w") as clustersfile:
        clusternumber, ncontigs = vamb.vambtools.write_clusters(
            clustersfile,
            maybe_split,
            max_clusters=maxclusters,
            min_size=minclustersize,
            rename=False,
            cluster_prefix=cluster_prefix
        )

    print("", file=logfile)
    log(f"Clustered {ncontigs} contigs in {clusternumber} bins", logfile, 1)

    elapsed = round(time.time()/60 - begintime, 2)
    log(f"Clustered contigs in {elapsed} minutes", logfile, 1)


def write_fasta(
    outdir: str,
    clusterspath: str,
    fastapath: str,
    contignames: np.ndarray,
    contiglengths: np.ndarray,
    minfasta: int,
    logfile: IO[str],
    separator: str,
) -> None:
    begintime = time.time()/60

    log("\nWriting FASTA files", logfile)
    log("Minimum FASTA size: "+str(minfasta), logfile, 1)
    assert len(contignames) == len(contiglengths)

    lengthof = dict(zip(contignames, contiglengths))
    filtered_clusters: dict[str, set[str]] = dict()

    with open(clusterspath) as file:
        clusters = vamb.vambtools.read_clusters(file)

    for cluster, contigs in clusters.items():
        size = sum(lengthof[contig] for contig in contigs)
        if size >= minfasta:
            filtered_clusters[cluster] = clusters[cluster]

    del lengthof, clusters
    keep: set[str] = set()
    for contigs in filtered_clusters.values():
        keep.update(set(contigs))
 
    with vamb.vambtools.Reader(fastapath) as file:
        vamb.vambtools.write_bins(
            os.path.join(outdir, "bins"), filtered_clusters, file, maxbins=None, separator=separator
        )

    ncontigs = sum(map(len, filtered_clusters.values()))
    nfiles = len(filtered_clusters)
    print("", file=logfile)
    log(f"Wrote {ncontigs} contigs to {nfiles} FASTA files", logfile, 1)

    elapsed = round(time.time()/60 - begintime, 2)
    log(f"Wrote FASTA in {elapsed} minutes", logfile, 1)


def run(
    outdir: str,
    fastapath: Optional[str],
    k: int,
    contrastive_vae: bool,
    contrastive_aae: bool,
    augmode: list[int],
    augdatashuffle: bool,
    augmentationpath: Optional[str],
    compositionpath: Optional[str],
    jgipath: Optional[str],
    bampaths: Optional[list[str]], 
    rpkmpath: Optional[str],
    mincontiglength: int,
    norefcheck: bool,
    noencode: bool,
    minid: float,
    minalignmentscore: float,
    vae_temperature: float,
    aae_temperature: float,
    nthreads: int,
    nhiddens: Optional[list[int]],
    nhiddens_aae: Optional[list[int]],
    nlatent: int,
    nlatent_aae_z: int,
    nlatent_aae_y: int,
    nepochs: int,
    nepochs_aae: int,
    batchsize: int,
    batchsize_aae: int,
    cuda: bool,
    alpha: Optional[float],
    beta: float,
    dropout: Optional[float],
    sl: float,
    slr: float,
    lrate_vae: float,
    lrate_aae: float,
    batchsteps: list[int],
    batchsteps_aae: list[int],
    windowsize: int,
    minsuccesses: int,
    minclustersize: int,
    separator: Optional[str],
    maxclusters: Optional[int],
    minfasta: Optional[int],
    model_selection: str,
    use_pc: bool,
    logfile: IO[str]
):

    contrastive = contrastive_aae or contrastive_vae
    if contrastive:
        log('Starting ClAVAmb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    else:
        log('Starting AVAmb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    log("Date and time is " + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()/60
    # Get TNFs, save as npz

    composition = calc_tnf(outdir,
                           fastapath,
                           compositionpath,
                           mincontiglength,
                           logfile,
                           nepochs,
                           augmode=augmode,
                           augmentation_store_dir=augmentationpath,
                           contrastive=contrastive,
                           k=k,
                           use_pc = use_pc)
    
    # Parse BAMs, save as npz
    refhash = None if norefcheck else vamb.vambtools._hash_refnames(composition.metadata.identifiers)
    abundance = calc_rpkm(
        outdir,
        bampaths,
        rpkmpath,
        jgipath,
        refhash,
        len(composition.metadata.identifiers),
        mincontiglength,
        minalignmentscore,
        minid,
        nthreads,
        logfile,
    )

    '''np.savez_compressed(os.path.join(outdir, "rpkm.npz"),
                        matrix=abundance,
                        samplenames=composition.samplenames,
                        minid=minid,
                        refhash=refhash)'''

    timepoint_gernerate_input=time.time()/60
    time_generating_input= round(timepoint_gernerate_input-begintime,2)

    assert len(abundance) == len(composition.matrix)

    if noencode:
        elapsed = round(time.time()/60 - begintime, 2)
        log(
            f"\nNoencode set, skipping encoding and clustering.\n\nCompleted Avamb in {elapsed} minutes",
            logfile,
        )
        return None
    log(f"\nTNF and coabundances generated in {time_generating_input} minutes", logfile, 1)

    # Estimate the number of clusters
    if nlatent_aae_y == None:
        log(f"\nEstimate the number of clusters", logfile, 1)
        begintime = time.time()/60
        #estimator = gmeans(np.concatenate((composition.matrix, abundance), axis=1))
        estimator = vamb.species_number.gmeans(abundance, logfile)
        estimator.process()
        nlatent_aae_y = len(estimator.get_clusters())
        timepoint_gernerate_input=time.time()/60
        time_generating_input= round(timepoint_gernerate_input-begintime,2)
        log(f"\nCluster estimated in {time_generating_input}", logfile, 1)
        log(f"\nEstimated {nlatent_aae_y} clusters", logfile, 1)
        exit()

    #Training phase
    if 'vae' in model_selection:
        begin_train_vae=time.time()/60
        # Train, save model
        mask, latent = trainvae(
            outdir,
            abundance,
            composition.matrix,
            k,
            contrastive_vae,
            augmode,
            augdatashuffle,
            augmentationpath,
            vae_temperature,
            composition.metadata.lengths,
            nhiddens,
            nlatent,
            alpha,
            beta,
            dropout,
            cuda,
            batchsize,
            nepochs,
            lrate_vae,
            batchsteps,
            logfile,
        )
        fin_train_vae=time.time()/60
        time_training_vae=round(fin_train_vae-begin_train_vae,2)
        log(f"\nVAE trained in {time_training_vae}", logfile, 1)
       
    if 'aae' in model_selection:
        begin_train_aae = time.time()/60
        # Train, save model
        mask, latent_z, clusters_y_dict = trainaae(
            outdir,
            abundance,
            composition.matrix,
            k,
            contrastive_aae,
            augmode,
            augdatashuffle,
            augmentationpath,
            aae_temperature,
            composition.metadata.lengths,
            nhiddens_aae,
            nlatent_aae_z,
            nlatent_aae_y,
            alpha,
            sl,
            slr,
            cuda,
            batchsize_aae,
            nepochs_aae,
            lrate_aae,
            batchsteps_aae,
            logfile,
            composition.metadata.identifiers,
        )
        fin_train_aae=time.time()/60
        time_training_aae=round(fin_train_aae-begin_train_aae,2)
        log(f"\nAAE trained in {time_training_aae}", logfile, 1)
     
    # Free up memory
    comp_metadata = composition.metadata
    del composition, abundance

    comp_metadata.filter_mask(mask)  # type: ignore
    # Write contignames and contiglengths needed for dereplication purposes 
    np.savetxt(os.path.join(outdir,'contignames'),comp_metadata.identifiers, fmt='%s')
    np.savez(os.path.join(outdir,'lengths.npz'),comp_metadata.lengths)
    
    if 'vae' in model_selection:
        assert comp_metadata.nseqs == len(latent)

        begin_cluster_latent=time.time()/60
        # Cluster, save tsv file
        clusterspath = os.path.join(outdir, "vae_clusters.tsv")
        cluster(
            clusterspath,
            latent,
            comp_metadata.identifiers,
            windowsize,
            minsuccesses,
            maxclusters,
            minclustersize,
            separator,
            cuda,
            logfile,
            'vae_',
        )
        fin_cluster_latent=time.time()/60
        time_clustering_latent=round(fin_cluster_latent-begin_cluster_latent,2)
        log(f"\nVAE latent clustered in {time_clustering_latent}", logfile, 1)

        del latent

        if minfasta is not None and fastapath is not None:
            # We have already checked fastapath is not None if minfasta is not None.
            write_fasta(
                outdir,
                clusterspath,
                fastapath,
                comp_metadata.identifiers,
                comp_metadata.lengths,
                minfasta,
                logfile,
                separator
            )

        writing_bins_time = round(time.time()/60 - fin_cluster_latent, 2)
        log(f"\nVAE bins written in {writing_bins_time} minutes", logfile)
            
        #log(f"\nCompleted Vamb in {elapsed} minutes", logfile)
    if 'aae' in model_selection:
        assert comp_metadata.nseqs == len(latent_z)

        begin_cluster_latent_z=time.time()/60
        # Cluster, save tsv file
        clusterspath = os.path.join(outdir, "aae_z_clusters.tsv")
        cluster(
            clusterspath,
            latent_z,
            comp_metadata.identifiers,
            windowsize,
            minsuccesses,
            maxclusters,
            minclustersize,
            separator,
            cuda,
            logfile,
            'aae_z_',
        )
        fin_cluster_latent_z=time.time()/60
        time_clustering_latent_z=round(fin_cluster_latent_z-begin_cluster_latent_z,2)
        log(f"\nAAE z latent clustered in {time_clustering_latent_z}", logfile, 1)

        del latent_z

        if minfasta is not None and fastapath is not None:
            # We have already checked fastapath is not None if minfasta is not None.
            write_fasta(
                outdir,
                clusterspath,
                fastapath,
                comp_metadata.identifiers,
                comp_metadata.lengths,
                minfasta,
                logfile,
                separator
            )
        time_writing_bins_z=time.time()/60
        writing_bins_time_z = round(time_writing_bins_z - fin_cluster_latent_z, 2)
        log(f"\nAAE z bins written in {writing_bins_time_z} minutes", logfile)
        
        clusterspath= os.path.join(outdir, "aae_y_clusters.tsv") 
         # Binsplit if given a separator
        if separator is not None:
            maybe_split = vamb.vambtools.binsplit(clusters_y_dict, separator)
        else:
            maybe_split = clusters_y_dict
        with open(clusterspath, "w") as clustersfile:
            clusternumber, ncontigs = vamb.vambtools.write_clusters(
                clustersfile,
                maybe_split,
                max_clusters=maxclusters,
                min_size=minclustersize,
                rename=False,
                cluster_prefix='aae_y_'
            )

        print("", file=logfile)
        log(f"Clustered {ncontigs} contigs in {clusternumber} bins", logfile, 1)
        time_start_writin_y_bins=time.time()/60
        if minfasta is not None and fastapath is not None:
            # We have already checked fastapath is not None if minfasta is not None.
            write_fasta(
                outdir,
                clusterspath,
                fastapath,
                comp_metadata.identifiers,
                comp_metadata.lengths,
                minfasta,
                logfile,
                separator
            )
        time_writing_bins_y=time.time()/60
        writing_bins_time_y = round(time_writing_bins_y - time_start_writin_y_bins , 2)
        log(f"\nAAE y bins written in {writing_bins_time_y} minutes", logfile)
      
def main():
    doc = f"""CL-Avamb: Contrastive Learning - Adversarial and Variational autoencoders for metagenomic binning.
    
    Version: {'.'.join([str(i) for i in vamb.__version__])}

    Default use, good for most datasets:
    vamb --outdir out --fasta my_contigs.fna --bamfiles *.bam -o C 

    For advanced use and extensions of CL-Avamb, check documentation of the package
    at https://github.com/99lorenzospina/CLAVAMB_DEC/tree/avamb_new."""
    parser = argparse.ArgumentParser(
        prog="vamb",
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s outdir tnf_input rpkm_input [options]",
        add_help=False,
    )

    # Help
    helpos = parser.add_argument_group(title="Help and version", description=None)
    helpos.add_argument("-h", "--help", help="print help and exit", action="help")
    helpos.add_argument(
        "--version",
        action="version",
        version=f'CL-Avamb {".".join(map(str, vamb.__version__))}',
    )

    # Positional arguments
    reqos = parser.add_argument_group(title="Output (required)", description=None)
    reqos.add_argument(
        "--outdir", metavar="", required=True, help="output directory to create"
    )

    # TNF arguments
    tnfos = parser.add_argument_group(
        title="TNF input (either fasta or all .npz files required)"
    )
    tnfos.add_argument("--fasta", metavar="", nargs='+', help="path to fasta file or paths to fasta files")
    tnfos.add_argument('--k', dest='k', metavar='', type=int, default=4, help='k for kmer calculation [4]')
    tnfos.add_argument("--composition", metavar="", help="path to .npz of composition")
    tnfos.add_argument("--use_pc", action='store_true', default=False, help='Wether to use pcmers instead of tnf [False]')

    # Contrastive learning arguments
    contrastiveos = parser.add_argument_group(title='Contrastive learning input')
    contrastiveos.add_argument('--contrastive_vae', action='store_true', default=False, help='Whether to perform contrastive learning(CLMB) or not(VAMB). [False]')
    contrastiveos.add_argument('--contrastive_aae', action='store_true', default=False, help='Whether to perform contrastive learning(CLAMB) or not(AAMB). [False]')
    contrastiveos.add_argument('--augmode', metavar='', nargs = 2, type = int, default=[3, 3],
                        help='The augmentation method. Requires 2 int. specify -1 if trying all augmentation methods. Choices: 0 for gaussian noise, 1 for transition, 2 for transversion, 3 for mutation, -1 for all. [3, 3]')
    contrastiveos.add_argument('--augdatashuffle', action='store_true', default=False,
            help='Whether to shuffle the training augmentation data (True: For each training, random select the augmentation data from the augmentation dir pool.\n!!!BE CAUTIOUS WHEN TURNING ON [False])')
    contrastiveos.add_argument('--augmentation', metavar='', help='path to augmentation dir. [outdir/augmentation]')
    contrastiveos.add_argument('--temperature', metavar='', default=1, type=float, help='The temperature for the normalized temperature-scaled cross entropy loss. [1]')
    
    # RPKM arguments
    rpkmos = parser.add_argument_group(
        title="RPKM input (either BAMs or .npz or .jgi required)"
    )
    rpkmos.add_argument(
        "--bamfiles", metavar="", help="paths to (multiple) BAM files", nargs="+"
    )
    rpkmos.add_argument("--rpkm", nargs="+", metavar="", help="path or paths to .npz of RPKM (abundances)")
    rpkmos.add_argument('--jgi', metavar='', help='path to output of jgi_summarize_bam_contig_depths')

    # Optional arguments
    inputos = parser.add_argument_group(title="IO options", description=None)

    inputos.add_argument(
        "-m",
        dest="minlength",
        metavar="",
        type=int,
        default=250,
        help="ignore contigs shorter than this [250]",
    )
    inputos.add_argument(
        "-z",
        dest="minid",
        metavar="",
        type=float,
        default=0.0,
        help="ignore reads with nucleotide identity below this [0.0]",
    )
    inputos.add_argument('-s', dest='minascore', metavar='', type=int, default=None,
                         help='ignore reads with alignment score below this [None]')
    inputos.add_argument(
        "-p",
        dest="nthreads",
        metavar="",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "number of threads to use " "[min(" + str(DEFAULT_THREADS) + ", nbamfiles)]"
        ),
    )
    inputos.add_argument(
        "--norefcheck",
        help="skip reference name hashing check [False]",
        action="store_true",
    )
    inputos.add_argument(
        "--minfasta",
        dest="minfasta",
        metavar="",
        type=int,
        default=None,
        help="minimum bin size to output as fasta [None = no files]",
    )
    inputos.add_argument(
        "--noencode",
        help="Output tnfs and abundances only, do not encode or cluster [False]",
        action="store_true",
    )
    
    # Model selection argument
    model_selection = parser.add_argument_group(title='Model selection', description=None)

    model_selection.add_argument('--model', dest='model', metavar='', type=str, default='vae&aae',
                         help='Choose which model to run; only vae (vae), only aae (aae), the combination of vae and aae (vae&aae), [vae&aae]')

    # VAE arguments
    vaeos = parser.add_argument_group(title="VAE options", description=None)

    vaeos.add_argument(
        "-n",
        dest="nhiddens",
        metavar="",
        type=int,
        nargs="+",
        default=None,
        help="hidden neurons [Auto]",
    )
    vaeos.add_argument(
        "-l",
        dest="nlatent",
        metavar="",
        type=int,
        default=32,
        help="latent neurons [32]",
    )
    vaeos.add_argument(
        "-a",
        dest="alpha",
        metavar="",
        type=float,
        default=None,
        help="alpha, weight of TNF versus depth loss [Auto]",
    )
    vaeos.add_argument(
        "-b",
        dest="beta",
        metavar="",
        type=float,
        default=200.0,
        help="beta, capacity to learn [200.0]",
    )
    vaeos.add_argument(
        "-d",
        dest="dropout",
        metavar="",
        type=float,
        default=None,
        help="dropout [Auto]",
    )
    vaeos.add_argument(
        "--cuda", help="use GPU to train & cluster [False]", action="store_true"
    )
    vaeos.add_argument('--v_temp', dest='vae_temperature', metavar='', type=float,
                        default=0.1596, help=' Temperature of the softcategorical prior [0.1596]')

    trainos = parser.add_argument_group(title="Training options", description=None)

    trainos.add_argument(
        "-e", dest="nepochs", metavar="", type=int, default=300, help="epochs [300]"
    )
    trainos.add_argument(
        "-t",
        dest="batchsize",
        metavar="",
        type=int,
        default=256,
        help="starting batch size [256]",
    )
    trainos.add_argument(
        "-q",
        dest="batchsteps",
        metavar="",
        type=int,
        nargs="*",
        default=[25, 75, 150, 225],
        help="double batch size at epochs [25 75 150 225]",
    )
    trainos.add_argument(
        "-r",
        dest="lrate_vae",
        metavar="",
        type=float,
        default=1e-3,
        help="learning rate [0.001]",
    )
    # AAE arguments
    aaeos = parser.add_argument_group(title='AAE options', description=None)

    aaeos.add_argument('--n_aae', dest='nhiddens_aae', metavar='', type=int, nargs='+',
                        default=None, help='hidden neurons AAE [Auto]')
    aaeos.add_argument('--z_aae', dest='nlatent_aae_z', metavar='', type=int,
                        default=283, help='latent neurons AAE continuous latent space  [283]')
    aaeos.add_argument('--y_aae', dest='nlatent_aae_y', metavar='', type=int,
                        default=None, help='latent neurons AAE categorical latent space [None]. If None, and estimation will be computed')
    aaeos.add_argument('--sl_aae', dest='sl', metavar='', type=float,
                        default=0.00964, help='loss scale between reconstruction loss and adversarial losses [0.00964] ')
    aaeos.add_argument('--slr_aae', dest='slr', metavar='', type=float,
                        default=0.5, help='loss scale between reconstruction adversarial losses [0.5] ')
    aaeos.add_argument('--a_temp', dest='aae_temperature', metavar='', type=float,
                        default=0.1596, help=' Temperature of the softcategorical prior [0.1596]')

    aaetrainos = parser.add_argument_group(title='Training options AAE', description=None)

    aaetrainos.add_argument('--e_aae', dest='nepochs_aae', metavar='', type=int,
                        default=70, help='epochs AAE [70]')
    aaetrainos.add_argument('--t_aae', dest='batchsize_aae', metavar='', type=int,
                        default=256, help='starting batch size AAE [256]')
    aaetrainos.add_argument('--q_aae', dest='batchsteps_aae', metavar='', type=int, nargs='*',
                        default=[25,50], help='double batch size at epochs AAE [25,50]')
    aaetrainos.add_argument('--r_aae', dest='lrate_aae',  metavar='',type=float,
                        default=1e-3, help='learning rate AAE [0.001]')

    # Clustering arguments
    clusto = parser.add_argument_group(title="Clustering options", description=None)
    clusto.add_argument(
        "-w",
        dest="windowsize",
        metavar="",
        type=int,
        default=200,
        help="size of window to count successes [200]",
    )
    clusto.add_argument(
        "-u",
        dest="minsuccesses",
        metavar="",
        type=int,
        default=20,
        help="minimum success in window [20]",
    )
    clusto.add_argument(
        "-i",
        dest="minsize",
        metavar="",
        type=int,
        default=1,
        help="minimum cluster size [1]",
    )
    clusto.add_argument(
        "-c",
        dest="maxclusters",
        metavar="",
        type=int,
        default=None,
        help="stop after c clusters [None = infinite]",
    )
    clusto.add_argument(
        "-o",
        dest="separator",
        metavar="",
        type=str,
        default=None,
        help="binsplit separator [None = no split]",
    )

    ######################### PRINT HELP IF NO ARGUMENTS ###################
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    outdir: str = os.path.abspath(args.outdir)
    fasta: Optional[list[str]] = args.fasta
    jgi: Optional[str] = args.jgi
    composition: Optional[str] = args.composition
    bamfiles: Optional[list[str]] = args.bamfiles
    rpkm: Optional[str] = args.rpkm
    minlength: int = args.minlength

    if args.minid != 0.0 and bamfiles is None:
        raise argparse.ArgumentTypeError(
            "If minid is set, RPKM must be passed as bam files"
        )

    minid: float = args.minid
    minalignmentscore : float = args.minascore
    nthreads: int = args.nthreads
    norefcheck: bool = args.norefcheck
    minfasta: Optional[int] = args.minfasta
    noencode: bool = args.noencode
    nhiddens: Optional[list[int]] = args.nhiddens
    
    nlatent: int = args.nlatent

    nhiddens_aae: Optional[list[int]] = args.nhiddens_aae
    nlatent_aae_z: int = args.nlatent_aae_z
    nlatent_aae_y: int = args.nlatent_aae_y


    alpha: Optional[float] = args.alpha
    beta: float = args.beta
    dropout: Optional[float] = args.dropout
    cuda: bool = args.cuda
    nepochs: int = args.nepochs

    nepochs_aae: int = args.nepochs_aae

    batchsize: int = args.batchsize
    batchsteps: list[int] = args.batchsteps

    batchsize_aae: int = args.batchsize_aae
    batchsteps_aae: list[int] = args.batchsteps_aae

    lrate_vae: float = args.lrate_vae
    vae_temperature: float = args.vae_temperature
    
    lrate_aae: float = args.lrate_aae
    aae_temperature: float = args.aae_temperature

    windowsize: int = args.windowsize
    minsuccesses: int = args.minsuccesses
    minsize: int = args.minsize
    maxclusters: Optional[int] = args.maxclusters
    separator: Optional[str] = args.separator

    contrastive: bool = (args.contrastive_vae) ^ (args.contrastive_aae)
    
    
    ######################### CHECK INPUT/OUTPUT FILES #####################

    # Outdir does not exist
    #if os.path.exists(outdir):
    #    raise FileExistsError(outdir)

    # Outdir is in an existing parent dir
    parentdir = os.path.dirname(outdir)
    if parentdir and not os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)

    # Make sure only one TNF input is there
    if not (composition is None) ^ (fasta is None):
        raise argparse.ArgumentTypeError(
            "Must specify either FASTA or composition path"
        )

    for path in (fasta, composition):
        if path is not None:
            if fasta is not None and not isinstance(fasta, str):    #if fasta has more paths, check all of them
                for p in fasta:
                    if not os.path.isfile(p):
                        raise FileNotFoundError(p)
            else:
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)

    # Check the running mode (CLMB or VAMB)
    if contrastive:
        if args.augmentation is None:
            augmentation_data_dir = os.path.join(args.outdir, 'augmentation')
        else:
            augmentation_data_dir = args.augmentation

        augmentation_number = [0, 0]
        aug_all_method = ['GaussianNoise','Transition','Transversion','Mutation','AllAugmentation']

        for i in range(2):
            if args.augmode[i] == -1:
                augmentation_number[i] = len(glob.glob(rf'{augmentation_data_dir+os.sep}pool{i}*k{args.k}*'))
            elif 0<= args.augmode[i] <= 3:
                augmentation_number[i] = len(glob.glob(rf'{augmentation_data_dir+os.sep}pool{i}*k{args.k}*_{aug_all_method[args.augmode[i]]}_*'))
            else:
                raise argparse.ArgumentTypeError('If contrastive learning is on, augmode must be int >-2 and <4')

        if fasta is None:
            warnings.warn("CLMB can't recognize the type of augmentation data, so please make sure your augmentation data in augmentation dir fit the augmode", UserWarning)
            if augmentation_number[0] == 0 or augmentation_number[1] == 0:
                raise argparse.ArgumentTypeError('Must specify either FASTA or the augmentation .npz inputs')
            if (2 * augmentation_number[0]) ** 2 < args.nepochs or (2 * augmentation_number[1]) ** 2 < args.nepochs:
                warnings.warn("Not enough augmentation, use replicated data in the training, which might decrease the performance", FutureWarning)
        else:
            if 0 < (2 * augmentation_number[0]) ** 2 < args.nepochs or 0 < (2 * augmentation_number[1]) ** 2 < args.nepochs:
                warnings.warn("Not enough augmentation, regenerate the augmentation to maintain the performance, please use ctrl+C to stop this process in 20 seconds if you would not like the augmentation dir to be rewritten. \
                    You can choose using the augmentations in the augmentation dir (without specifying --fasta) after interrupting this program, or continuing this program to erase the augmentation dir and regenerate the augmentation data", UserWarning)
                for sleep_time in range(4):
                    print(f'Program to be continued in {20-4*sleep_time}s, please use ctrl+C to stop this process if you would not like the augmentation dir to be rewritten')
                    time.sleep(5)
                warnings.warn("Not enough augmentation, regenerate the augmentation to maintain the performance, erasing the augmentation dir. We will regenerate the augmentation in the following function", UserWarning)
                for erase_file in glob.glob(rf'{augmentation_data_dir+os.sep}pool*k{args.k}*'):
                    print(f'removing {erase_file} ...')
                    os.system(f'rm {erase_file}')

    else:
        augmentation_data_dir = os.path.join(args.outdir, 'augmentation')
    
    # Make sure only one RPKM input is there
    if not( (bamfiles is not None and rpkm is None and jgi is None) or \
   (bamfiles is None and rpkm is not None and jgi is None) or \
   (bamfiles is None and rpkm is None and jgi is not None)):
        raise argparse.ArgumentTypeError(
            "Must specify exactly one of BAM files, JGI file or RPKM input"
        )

    if rpkm is not None:
        if isinstance(rpkm, str):
            if not os.path.isfile(rpkm):
                raise FileNotFoundError("Not an existing non-directory file: " + rpkm)
        else:
            for r in rpkm:
                if not os.path.isfile(r):
                    raise FileNotFoundError("Not an existing non-directory file: " + r)

    if bamfiles is not None:
        for bampath in bamfiles:
            if not os.path.isfile(bampath):
                raise FileNotFoundError(
                    "Not an existing non-directory file: " + bampath
                )

            # Check this early, since I expect users will forget about this
            if not vamb.parsebam.is_bam_sorted(bampath):
                raise ValueError(f"BAM file {bampath} is not sorted by reference.")

    # Check minfasta settings
    if minfasta is not None and fasta is None:
        raise argparse.ArgumentTypeError(
            "If minfasta is not None, " "input fasta file must be given explicitly"
        )

    if minfasta is not None and minfasta < 0:
        raise argparse.ArgumentTypeError(
            "Minimum FASTA output size must be nonnegative"
        )

    ####################### CHECK ARGUMENTS FOR TNF AND BAMFILES ###########
    if minlength < 250:
        raise argparse.ArgumentTypeError("Minimum contig length must be at least 250")

    if not isfinite(minid) or minid < 0.0 or minid > 1.0:
        raise argparse.ArgumentTypeError("Minimum nucleotide ID must be in [0,1]")

    if nthreads < 1:
        raise argparse.ArgumentTypeError("Zero or negative subprocesses requested")

    ####################### CHECK VAE/AAE OPTIONS ################################
    if nhiddens is not None and any(i < 1 for i in nhiddens):
        raise argparse.ArgumentTypeError(
            f"Minimum 1 neuron per layer, not {min(nhiddens)}"
        )

    if nhiddens_aae is not None and any(i < 1 for i in nhiddens_aae):
        raise argparse.ArgumentTypeError(
            f"Minimum 1 neuron per layer, not {min(nhiddens_aae)}"
        )

    if nlatent < 1:
        raise argparse.ArgumentTypeError(f"Minimum 1 latent neuron, not {nlatent}")
    
    if nlatent_aae_z < 1:
        raise argparse.ArgumentTypeError(f"Minimum 1 latent neuron, not {nlatent_aae_z}")

    if alpha is not None and (alpha <= 0 or alpha >= 1):
        raise argparse.ArgumentTypeError("alpha must be above 0 and below 1")

    if beta <= 0:
        raise argparse.ArgumentTypeError("beta cannot be negative or zero")

    if dropout is not None and (dropout < 0 or dropout >= 1):
        raise argparse.ArgumentTypeError("dropout must be in 0 <= d < 1")

    if cuda and not torch.cuda.is_available():
        raise ModuleNotFoundError("Cuda is not available on your PyTorch installation")

    ###################### CHECK TRAINING OPTIONS ####################
    if nepochs < 1:
        raise argparse.ArgumentTypeError(f"Minimum 1 epoch, not {nepochs}")

    if batchsize < 1:
        raise argparse.ArgumentTypeError(f"Minimum batchsize of 1, not {batchsize}")

    batchsteps = sorted(set(batchsteps))
    if max(batchsteps, default=0) >= nepochs:
        raise argparse.ArgumentTypeError("All batchsteps must be less than nepochs")

    if min(batchsteps, default=1) < 1:
        raise argparse.ArgumentTypeError("All batchsteps must be 1 or higher")

    if lrate_vae <= 0 or lrate_aae <= 0:
        raise argparse.ArgumentTypeError("Learning rate must be positive")

    ###################### CHECK CLUSTERING OPTIONS ####################
    if minsize < 1:
        raise argparse.ArgumentTypeError("Minimum cluster size must be at least 0")

    if windowsize < 1:
        raise argparse.ArgumentTypeError("Window size must be at least 1")

    if minsuccesses < 1 or minsuccesses > windowsize:
        raise argparse.ArgumentTypeError("Minimum cluster size must be in 1:windowsize")

    if separator is not None and len(separator) == 0:
        raise argparse.ArgumentTypeError("Binsplit separator cannot be an empty string")

    ###################### SET UP LAST PARAMS ############################

    # This doesn't actually work, but maybe the PyTorch folks will fix it sometime.
    torch.set_num_threads(nthreads)

    ################### RUN PROGRAM #########################
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass
    except:
        raise
    if "aae" in args.model:
        try:
            os.mkdir(os.path.join(outdir,'tmp')) # needed for dereplication logs and files
        except FileExistsError:
            pass
        except:
            raise
        try:
            os.mkdir(os.path.join(outdir,'tmp','ripped_bins')) # needed for dereplication logs and files
        except FileExistsError:
            pass
        except:
            raise
        try:
            os.mkdir(os.path.join(outdir,'tmp','checkm2_all')) # needed for dereplication logs and files
        except FileExistsError:
            pass
        except:
            raise
        try:
            os.mkdir(os.path.join(outdir,'NC_bins')) # needed for dereplication logs and files
        except FileExistsError:
            pass
        except:
            raise


    logpath = os.path.join(outdir, "log.txt")

    with open(logpath, "w") as logfile:
        if args.contrastive_vae and args.contrastive_aae and nepochs_aae != nepochs:
            #because of augmented data generation technique, keeping two values
            #may create conflicts
            log('Setting same number of epochs for aae and vae', logfile)
            nepochs_aae = nepochs
        run(
            outdir,
            fasta,
            args.k,
            args.contrastive_vae,
            args.contrastive_aae,
            args.augmode,
            args.augdatashuffle,
            augmentation_data_dir,
            composition,
            jgi,
            bamfiles,
            rpkm,
            mincontiglength=minlength,
            norefcheck=norefcheck,
            noencode=noencode,
            minid=minid,
            minalignmentscore=minalignmentscore,
            vae_temperature = vae_temperature,
            aae_temperature = aae_temperature,
            nthreads=nthreads,
            nhiddens=nhiddens,
            nhiddens_aae=nhiddens_aae,
            nlatent=nlatent,
            nlatent_aae_z=nlatent_aae_z,
            nlatent_aae_y=nlatent_aae_y,
            nepochs=nepochs,
            nepochs_aae=nepochs_aae,
            batchsize=batchsize,
            batchsize_aae=batchsize_aae,
            cuda=cuda,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            sl=args.sl,
            slr=args.slr,
            lrate_vae=lrate_vae,
            lrate_aae=lrate_aae,
            batchsteps=batchsteps,
            batchsteps_aae=batchsteps_aae,
            windowsize=windowsize,
            minsuccesses=minsuccesses,
            minclustersize=minsize,
            separator=separator,
            maxclusters=maxclusters,
            minfasta=minfasta,
            model_selection=args.model,
            use_pc = args.use_pc,
            logfile=logfile,
        )


if __name__ == "__main__":
    main()

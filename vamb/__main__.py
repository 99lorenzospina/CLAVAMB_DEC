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
from math import isfinite
from typing import Optional, IO

_ncpu = os.cpu_count()
DEFAULT_THREADS = 8 if _ncpu is None else min(_ncpu, 8)

LOAD_MOD = False

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
    use_pc=False,
    use_tnf=True
) -> vamb.parsecontigs.Composition:
    begintime = time.time()/60
    log("\nLoading TNF/PC", logfile, 0)
    log(f"Minimum sequence length: {mincontiglength}", logfile, 1)
    if use_pc:
        log(f"Using pcmers", logfile, 1)
    if use_tnf:
        log(f"Using kmers", logfile, 1)
    if npzpath is not None:
        log(f"Loading composition from npz {npzpath}", logfile, 1)
        composition = vamb.parsecontigs.Composition.load(npzpath)
        composition.filter_min_length(mincontiglength)
    else:
        assert fastapath is not None
        if len(fastapath) == 1:
            log(f"Loading data from FASTA file {fastapath[0]}", logfile, 1)
            with vamb.vambtools.Reader(fastapath[0]) as file:
                composition = vamb.parsecontigs.Composition.from_file(
                    file, minlength=mincontiglength, use_pc = use_pc, use_tnf=use_tnf,
                )
                file.close()
            composition.save(os.path.join(outdir, "composition.npz"))
        else:        #multiple files in input (multiple datasets)
            log(f"Loading data from FASTA files {fastapath}", logfile, 1)
            b = True
            for path in fastapath:
                    with vamb.vambtools.Reader(path) as file:
                        if b:
                            b = False
                            composition = None
                        composition = vamb.parsecontigs.Composition.concatenate(composition, vamb.parsecontigs.Composition.from_file(
                            file, minlength=mincontiglength, use_pc=use_pc, use_tnf=use_tnf
                        ))
                        file.close()
            composition.save(os.path.join(outdir, "composition.npz"))
    
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
        if len(rpkmpath) == 1:
            log('Loading RPKM from npz array {}'.format(rpkmpath[0]), logfile, 1)
            rpkms = vamb.vambtools.read_npz(rpkmpath[0])
            #rpkms = vamb.parsebam.avg_window(rpkms)    #only if I want to keep the size 10 for every dataset
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

def traindec(
    outdir: str,
    rpkms: np.ndarray,
    tnfs: np.ndarray,
    lengths: np.ndarray,
    nhiddens: Optional[list[int]],  # set automatically if None
    nlatent_y: int,
    lr: float,  # set automatically if None
    cri_lr: float,
    dis_lr: float,
    max_iter: int,
    max_iter_pretrain: int,
    aux_iter: int,
    max_iter_dis: int,
    targ_iter: int,
    tol: float,
    cuda: bool,
    batchsize: int,
    lrate: float,
    logfile: IO[str],
    contignames: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict()]:

    begintime = time.time()/60
    log("\nCreating and training AAE", logfile)
    nsamples = rpkms.shape[1]    #number of contigs

    
    assert len(rpkms) == len(tnfs)

    dataloader = vamb.encode.make_dataloader(
        rpkms, tnfs, lengths, batchsize, destroy=True, cuda=cuda, shuffle=True
    )
    log("Created dataloader", logfile, 1)
    log(f"Number of sequences used: {len(rpkms)}", logfile, 1)
    print("", file=logfile)

    y_pred = None

    if not LOAD_MOD:
        aaedec = vamb.aambdec_encode.AAEDEC(ntnf=int(tnfs.shape[1]), nsamples=nsamples, nhiddens=nhiddens, nlatent_y=nlatent_y, lr=lr, cri_lr=cri_lr, dis_lr=dis_lr, _cuda=cuda)
        log("Created AAE", logfile, 1)
        modelpath = os.path.join(outdir, 'aaedec_model.pt')
        aaedec.pretrain(dataloader, max_iter_pretrain, logfile)
        y_pred = aaedec.trainmodel(dataloader, max_iter, aux_iter, max_iter_dis, targ_iter, tol, lrate, modelfile=modelpath, logfile=logfile)
    else:   #NOT WORKING: I should save also former y_preds and start retraining and also updating y_pred, dunno wether to do it
        modelpath = os.path.join(outdir, 'aaedec_model.pt')
        log("Loaded AAE", logfile, 1)
        aaedec = vamb.aambdec_encode.AAEDEC.load(modelpath,cuda=cuda,c=False)
        aaedec.to(('cuda' if cuda else 'cpu'))
    
    print("", file=logfile)
    log("Creating clusters", logfile, 1)
    clusters_y_dict = aaedec.get_dict(contignames, y_pred)
    del aaedec  # Needed to free "latent" array's memory references?

    elapsed = round(time.time()/60 - begintime, 2)
    log(f"Trained AAE and clustering in {elapsed} minutes", logfile, 1)

    return clusters_y_dict

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
    
    if len(fastapath) == 1:
        fastapath = fastapath[0]    #if I gave more dataset in input, I have to create the else
    
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

def apply_mask(composition, abundance, mask, logfile=None):
    begintime = time.time()/60
    data_list = []

    # Read the file and extract the content
    with open(mask, 'r') as file:
        content = file.read().strip()

    # Remove square brackets and split the content by commas
    content = content.replace('[', '').replace(']', '').replace("'", '')
    data_list = content.split(', ')

    # Remove the newline character from the last element
    data_list[-1] = data_list[-1].rstrip('\n')
    if logfile is not None:
        print(f"\nApplying mask of {len(data_list)} contigs", file=logfile)

    # Create a copy of composition.metadata.identifiers and composition.metadata.lengths
    identifiers_copy = composition.metadata.identifiers[:]
    lengths_copy = composition.metadata.lengths[:]
    mask_copy = composition.metadata.mask[:]
    matrix_copy = composition.matrix[:]
    minlength = composition.metadata.minlength
    l = len(lengths_copy)
    del composition

    # Find indices of elements to remove
    indices = [identifiers_copy[item] for item in data_list]
    # Remove elements from identifiers_copy and composition_metadata_lengths
    identifiers_copy = [identifiers_copy[i] for i in range(l) if i not in indices]
    lengths_copy = [lengths_copy[i] for i in range(l) if i not in indices]
    for i in indices:
        mask_copy[i] = False
    matrix_copy = np.delete(matrix_copy, indices, axis=0)

    abundance = np.delete(abundance, indices, axis=0)
    composition = vamb.parsecontigs.Composition(vamb.parsecontigs.CompositionMetaData(
                    identifiers_copy, lengths_copy, mask_copy, minlength), matrix_copy)

    timepoint_gernerate_input=time.time()/60
    time_generating_input= round(timepoint_gernerate_input-begintime,2)   

    if logfile is not None:
        print(f"\nMask applied in {time_generating_input} minutes", file=logfile)
        print(f"\t{len(identifiers_copy)} contigs have remained", file=logfile, end="\n\n")
    return composition, abundance

def run(
    outdir: str,
    fastapath: Optional[str],
    compositionpath: Optional[str],
    jgipath: Optional[str],
    bampaths: Optional[list[str]], 
    rpkmpath: Optional[str],
    mask: Optional[str],
    mincontiglength: int,
    norefcheck: bool,
    noencode: bool,
    minid: float,
    minalignmentscore: float,
    nthreads: int,
    nhiddens_aae: Optional[list[int]],
    nlatent_aae_y: int,
    lrate: float,
    lr: float,
    cri_lr: float,
    dis_lr: float,
    max_iter: int,
    max_iter_pretrain: int,
    aux_iter: int,
    max_iter_dis: int,
    targ_iter: int,
    tol: float,
    batchsize: int,
    cuda: bool,
    minclustersize: int,
    separator: Optional[str],
    maxclusters: Optional[int],
    minfasta: Optional[int],
    model_selection: str,
    use_pc: bool,
    use_tnf: bool,
    logfile: IO[str]
):

    log('Starting AVAmb version ' + '.'.join(map(str, vamb.__version__)), logfile)
    log("Date and time is " + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()/60
    # Get TNFs, save as npz

    composition = calc_tnf(outdir,
                           fastapath,
                           compositionpath,
                           mincontiglength,
                           logfile,
                           use_pc = use_pc,
                           use_tnf = use_tnf)
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
    log(f"\nTNF/PC and coabundances generated in {time_generating_input} minutes", logfile, 1)

    composition, abundance = apply_mask(composition, abundance, mask, logfile)

    # Estimate the number of clusters
    if 'aaedec' in model_selection and nlatent_aae_y == None:
        log(f"\nEstimate the number of clusters", logfile, 1)
        begintime = time.time()/60
        #estimator = gmeans(np.concatenate((composition.matrix, abundance), axis=1))
        estimator = vamb.species_number.gmeans(np.concatenate((composition.matrix, abundance), axis=1), logfile, ccore = False)
        estimator.process()
        nlatent_aae_y = len(estimator.get_definitive_centers())
        timepoint_gernerate_input=time.time()/60
        time_generating_input= round(timepoint_gernerate_input-begintime,2)
        log(f"\nCluster estimated in {time_generating_input} minutes", logfile, 1)
        log(f"\nEstimated {nlatent_aae_y} clusters", logfile, 1)

    if 'aaedec' in model_selection:
        begin_train_aae = time.time()/60
        # Train, save model
        clusters_y_dict = traindec(
            outdir,
            abundance,
            composition.matrix,
            composition.metadata.lengths,
            nhiddens_aae,
            nlatent_aae_y,
            lr,
            cri_lr,
            dis_lr,
            max_iter,
            max_iter_pretrain,
            aux_iter,
            max_iter_dis,
            targ_iter,
            tol,
            cuda,
            batchsize,
            lrate,
            logfile,
            composition.metadata.identifiers,
        )
        fin_train_aae=time.time()/60
        time_training_aae=round(fin_train_aae-begin_train_aae,2)
        log(f"\nAAE trained in {time_training_aae}", logfile, 1)
     
    # Free up memory
    comp_metadata = composition.metadata
    del composition, abundance

    # Write contignames and contiglengths needed for dereplication purposes 
    np.savetxt(os.path.join(outdir,'contignames'),comp_metadata.identifiers, fmt='%s')
    np.savez(os.path.join(outdir,'lengths.npz'),comp_metadata.lengths)

    if 'aaedec' in model_selection:        
        clusterspath= os.path.join(outdir, "aaedec_clusters.tsv") 

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
                cluster_prefix='aaedec_'
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
        log(f"\nAAEDEC bins written in {writing_bins_time_y} minutes", logfile)
      
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
    tnfos.add_argument("--composition", metavar="", help="path to .npz of composition")
    tnfos.add_argument("--use_pc", action='store_true', default=False, help='Wether to use pcmers for composition [False]')
    tnfos.add_argument("--use_tnf", action='store_true', default=False, help='Wether to use tnf for composition [False]')
    
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
        default=100,
        help="ignore contigs shorter than this [100]",
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
    inputos.add_argument('--mask', dest='mask', metavar="", help="contigs already clustered")
    
    # Model selection argument
    model_selection = parser.add_argument_group(title='Model selection', description=None)

    model_selection.add_argument('--model', dest='model', metavar='', type=str, default='aaedec',
                         help='Choose which model to run; ADEC (aaedec), [aaedec]')

    # AAE arguments
    aaeos = parser.add_argument_group(title='AAE options', description=None)

    aaeos.add_argument('--n_aae', dest='nhiddens_aae', metavar='', type=int, nargs='+',
                        default=None, help='hidden neurons AAE [Auto]')
    aaeos.add_argument('--y_aae', dest='nlatent_aae_y', metavar='', type=int,
                        default=None, help='latent neurons AAE categorical latent space [None]. If None, and estimation will be computed')
    aaeos.add_argument('--lr_aae', dest='lr', metavar='', type=float,
                        default=1e-3, help='learning rate for pretraining [0.001]')
    aaeos.add_argument('--cri_lr_aae', dest='cri_lr', metavar='', type=float,
                        default=1e-3, help='learning rate for critic [0.001]')
    aaeos.add_argument('--dis_lr_aae', dest='dis_lr', metavar='', type=float,
                        default=1e-3, help='learning rate for discriminator [0.001]')
    aaeos.add_argument(
        "--cuda", help="use GPU to train & cluster [False]", action="store_true", default=False
    )
    aaeos.add_argument(
        "-d",
        dest="dropout",
        metavar="",
        type=float,
        default=None,
        help="dropout [Auto]",
    )

    aaetrainos = parser.add_argument_group(title='Training options AAE', description=None)

    aaetrainos.add_argument('--lrate_aae', dest='lrate',  metavar='',type=float,
                        default=1e-4, help='learning rate for AAEDEC training [0.0001]')
    aaetrainos.add_argument('--max_iter_aae', dest='max_iter',  metavar='',type=int,
                        default=300, help='maximum number of iterations in training [300]')
    aaetrainos.add_argument('--max_iter_dis_aae', dest='max_iter_dis',  metavar='',type=int,
                        default=200, help='maximum number of iterations in discriminator pretraining [200]')
    aaetrainos.add_argument('--aux_iter_aae', dest='aux_iter',  metavar='',type=int,
                        default=10, help='auxiliary threshold in training [10]')
    aaetrainos.add_argument('--targ_iter_aae', dest='targ_iter',  metavar='',type=int,
                        default=10, help='batchstep update [10]')
    aaetrainos.add_argument('--max_iter_pretrain_aae', dest='max_iter_pretrain',  metavar='',type=int,
                        default=2000, help='maximum number of iterations in critic training [2000]')
    aaetrainos.add_argument('--tol_aae', dest='tol',  metavar='',type=float,
                        default=1e-6, help='tollerance threshold [0.0001%]')
    aaetrainos.add_argument(
        "-t",
        dest="batchsize",
        metavar="",
        type=int,
        default=256,
        help="starting batch size [256]",
    )

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
    mask: Optional[str] = args.mask

    nhiddens_aae: Optional[list[int]] = args.nhiddens_aae
    nlatent_aae_y: int = args.nlatent_aae_y

    dropout: Optional[float] = args.dropout
    cuda: bool = args.cuda

    batchsize: int = args.batchsize
    
    lrate: float = args.lrate

    windowsize: int = args.windowsize
    minsuccesses: int = args.minsuccesses
    minsize: int = args.minsize
    maxclusters: Optional[int] = args.maxclusters
    separator: Optional[str] = args.separator

    use_tnf = False
    if args.use_tnf:
        use_tnf = True
    else:
        if not args.use_pc:
            use_tnf = True
    
    
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
    if minlength < 100:
        raise argparse.ArgumentTypeError("Minimum contig length must be at least 100")

    if not isfinite(minid) or minid < 0.0 or minid > 1.0:
        raise argparse.ArgumentTypeError("Minimum nucleotide ID must be in [0,1]")

    if nthreads < 1:
        raise argparse.ArgumentTypeError("Zero or negative subprocesses requested")

    ####################### CHECK AAEDEC OPTIONS ################################

    if nhiddens_aae is not None and any(i < 1 for i in nhiddens_aae):
        raise argparse.ArgumentTypeError(
            f"Minimum 1 neuron per layer, not {min(nhiddens_aae)}"
        )

    if dropout is not None and (dropout < 0 or dropout >= 1):
        raise argparse.ArgumentTypeError("dropout must be in 0 <= d < 1")

    if cuda and not torch.cuda.is_available():
        raise ModuleNotFoundError("Cuda is not available on your PyTorch installation")

    ###################### CHECK TRAINING OPTIONS ####################

    if batchsize < 1:
        raise argparse.ArgumentTypeError(f"Minimum batchsize of 1, not {batchsize}")

    if lrate <= 0 or args.lr <= 0 or args.cri_lr <= 0 or args.dis_lr <= 0:
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

    logpath = os.path.join(outdir, "log.txt")

    with open(logpath, "w") as logfile:
        run(
            outdir,
            fasta,
            composition,
            jgi,
            bamfiles,
            rpkm,
            mask=mask,
            mincontiglength=minlength,
            norefcheck=norefcheck,
            noencode=noencode,
            minid=minid,
            minalignmentscore=minalignmentscore,
            nthreads=nthreads,
            nhiddens_aae=nhiddens_aae,
            nlatent_aae_y=nlatent_aae_y,
            lrate=lrate,
            lr=args.lr,
            cri_lr=args.cri_lr,
            dis_lr=args.dis_lr,
            max_iter=args.max_iter,
            max_iter_pretrain = args.max_iter_pretrain,
            aux_iter=args.aux_iter,
            max_iter_dis=args.max_iter_dis,
            targ_iter=args.targ_iter,
            tol=args.tol,
            batchsize=batchsize,
            cuda=cuda,
            minclustersize=minsize,
            separator=separator,
            maxclusters=maxclusters,
            minfasta=minfasta,
            model_selection=args.model,
            use_pc = args.use_pc,
            use_tnf = use_tnf,
            logfile=logfile,
        )


if __name__ == "__main__":
    main()

__doc__ = """Estimate depths from BAM files of reads mapped to contigs.

Usage:
>>> bampaths = ['/path/to/bam1.bam', '/path/to/bam2.bam', '/path/to/bam3.bam']
>>> rpkms = Abundance.from_file(bampaths, metadata, True, 0.1, 3)
"""

import pycoverm
import os as _os
import numpy as _np
from math import isfinite
from vamb.parsecontigs import CompositionMetaData
import vamb.vambtools as _vambtools
from typing import Optional, TypeVar, Union, IO, Sequence

_ncpu = _os.cpu_count()
DEFAULT_THREADS = 8 if _ncpu is None else _ncpu

A = TypeVar("A", bound="Abundance")


class Abundance:
    "Object representing contig abundance. Contains a matrix and refhash."

    __slots__ = ["matrix", "samplenames", "minid", "refhash"]

    def __init__(
        self,
        matrix: _np.ndarray,
        samplenames: Sequence[str],
        minid: float,
        refhash: bytes,
    ):
        assert matrix.dtype == _np.float32
        assert matrix.ndim == 2
        assert matrix.shape[1] == len(samplenames)
        assert isfinite(minid) and 0.0 <= minid and minid <= 1.0

        self.matrix = matrix
        self.samplenames = _np.array(samplenames, dtype=object)
        self.minid = minid
        self.refhash = refhash

    @property
    def nseqs(self) -> int:
        return len(self.matrix)

    @property
    def nsamples(self) -> int:
        return len(self.samplenames)

    def verify_refhash(self, refhash: bytes) -> None:
        if self.refhash != refhash:
            raise ValueError(
                f"BAM files reference name hash to {self.refhash.hex()}, "
                f"expected {refhash.hex()}. "
                "Make sure all BAM and FASTA headers are identical "
                "and in the same order."
            )

    def save(self, io: Union[str, IO[bytes]]):
        _np.savez_compressed(
            io,
            matrix=self.matrix,
            samplenames=self.samplenames,
            minid=self.minid,
            refhash=self.refhash,
        )

    @classmethod
    def load(cls: type[A], io: Union[str, IO[bytes]], refhash: Optional[bytes]) -> A:
        arrs = _np.load(io, allow_pickle=True)
        abundance = cls(
            _vambtools.validate_input_array(arrs["matrix"]),
            arrs["samplenames"],
            arrs["minid"].item(),
            arrs["refhash"].item(),
        )
        if refhash is not None:
            abundance.verify_refhash(refhash)

        return abundance

    @classmethod
    def from_files(
        cls: type[A],
        paths: list[str],
        comp_metadata: CompositionMetaData,
        verify_refhash: bool,
        minid: float,
        nthreads: int,
    ) -> A:
        """Input:
        paths: List of paths to BAM files
        comp_metadata: CompositionMetaData of sequence catalogue used to make BAM files
        verify_refhash: Whether to verify composition and BAM references are the same
        minid: Discard any reads with nucleotide identity less than this
        nthreads: Use this number of threads for coverage estimation
        """
        if minid < 0 or minid > 1:
            raise ValueError(f"minid must be between 0 and 1, not {minid}")

        for path in paths:
            if not _os.path.isfile(path):
                raise FileNotFoundError(path)

            if not pycoverm.is_bam_sorted(path):
                raise ValueError(f"Path {path} is not sorted by reference.")

        # Workaround: Currently pycoverm has a bug where it filters contigs when mindid == 0
        # (issue #7). Can be solved by setting it to a low value
        _minid = minid if minid > 0.001 else 0.001
        headers, coverage = pycoverm.get_coverages_from_bam(
            paths,
            threads=nthreads,
            min_identity=_minid,
            # Note: pycoverm's trim_upper=0.1 is same as CoverM trim-upper 90.
            trim_upper=0.1,
            trim_lower=0.1,
        )

        assert len(headers) == len(coverage)
        assert coverage.shape[1] == len(paths)

        # Filter length, using comp_metadata's mask, which has been set by minlength
        if len(comp_metadata.mask) != len(headers):
            raise ValueError(
                f"CompositionMetaData was created with {len(comp_metadata.mask)} sequences, "
                f"but number of refs in BAM files are {len(headers)}."
            )

        headers = [h for (h, m) in zip(headers, comp_metadata.mask) if m]
        _vambtools.numpy_inplace_maskarray(coverage, comp_metadata.mask)

        refhash = _vambtools.hash_refnames(headers)
        abundance = cls(coverage, paths, minid, refhash)

        # Check refhash
        if verify_refhash:
            abundance.verify_refhash(comp_metadata.refhash)

        return abundance

def calc_rpkm(
    outdir: str,
    bampaths: Optional[list[str]],
    npzpath: Optional[str],
    jgipath: Optional[str],
    comp_metadata: vamb.parsecontigs.CompositionMetaData,
    verify_refhash: bool,
    minid: float,
    nthreads: int,
    logfile: IO[str],
) -> vamb.parsebam.Abundance:

    begintime = time.time()/60
    log("\nLoading depths", logfile)
    log(
        f'Reference hash: {comp_metadata.refhash.hex() if verify_refhash else "None"}',
        logfile,
        1,
    )

    # If rpkm is given, we load directly from .npz file
    if npzpath is not None:
        log(f"Loading depths from npz array {npzpath}", logfile, 1)
        abundance = vamb.parsebam.Abundance.load(
            npzpath, comp_metadata.refhash if verify_refhash else None
        )
        # I don't want this check in any constructors of abundance, since the constructors
        # should be able to skip this check in case comp and abundance are independent.
        # But when running the main Vamb workflow, we need to assert this.
        if abundance.nseqs != comp_metadata.nseqs:
            assert not verify_refhash
            raise ValueError(
                f"Loaded abundance has {abundance.nseqs} sequences, "
                f"but composition has {comp_metadata.nseqs}."
            )
    elif jgipath is not None:
        log('Loading RPKM from JGI file {}'.format(jgipath), logfile, 1)
        with open(jgipath) as file:
            rpkms = vamb.vambtools.load_jgi(file, comp_metadata.minlength, comp_metadata.refhash if verify_refhash else None)
            abundance = vamb.parsebam.Abundance(rpkms, [jgipath], minid, comp_metadata.refhash if verify_refhash else None)
    else:
        assert bampaths is not None
        log(f"Parsing {len(bampaths)} BAM files with {nthreads} threads", logfile, 1)

        abundance = vamb.parsebam.Abundance.from_files(
            bampaths, comp_metadata, verify_refhash, minid, nthreads
        )
        abundance.save(os.path.join(outdir, "abundance.npz"))

    log(f"Min identity: {abundance.minid}\n", logfile, 1)
    log("Order of columns is:", logfile, 1)
    log("\n\t".join(abundance.samplenames), logfile, 1)

    elapsed = round(time.time()/60 - begintime, 2)
    print("", file=logfile)
    log(f"Processed RPKM in {elapsed} minutes", logfile, 1)

    return abundance

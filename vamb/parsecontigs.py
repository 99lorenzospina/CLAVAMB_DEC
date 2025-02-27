__doc__ = """Calculate tetranucleotide frequency from a FASTA file.

Usage:
>>> with open('/path/to/contigs.fna') as filehandle
...     tnfs, contignames, lengths = read_contigs(filehandle)
"""
import pyximport
pyximport.install()
import vamb._vambtools as v
import os as _os
import numpy as _np
import vamb.vambtools as _vambtools
import vamb.mimics as mimics
from collections.abc import Iterable, Sequence
from typing import IO, Union, TypeVar
import math
import random
import time

# This kernel is created in src/create_kernel.py. See that file for explanation
_KERNEL: _np.ndarray = _vambtools.read_npz(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "kernel.npz")
)


class CompositionMetaData:
    """A class containing metadata of sequence composition.
    Current fields are:
    * identifiers: A Numpy array of objects, str identifiers of kept sequences
    * lengths: A Numpy vector of 32-bit uint lengths of kept sequences
    * mask: A boolean Numpy vector of which sequences were kept in original file
    * refhash: A bytes object representing the hash of the identifiers
    * minlength: The minimum contig length used for filtering
    """

    __slots__ = ["identifiers", "lengths", "mask", "refhash", "minlength"]

    def __init__(
        self,
        identifiers: _np.ndarray,
        lengths: _np.ndarray,
        mask: _np.ndarray,
        minlength: int,
    ):
        assert len(identifiers) == len(lengths)
        assert identifiers.dtype == _np.dtype("O")
        assert _np.issubdtype(lengths.dtype, _np.integer)
        assert mask.dtype == bool
        assert mask.sum() == len(lengths)
        assert lengths.min(initial=minlength) >= minlength

        if len(set(identifiers)) < len(identifiers):
            raise ValueError(
                "Sequence names must be unique, but are not. "
                "Vamb only uses the identifier (e.g. header before whitespace) as "
                "sequence identifiers. Verify identifier uniqueness."
            )

        self.identifiers = identifiers
        self.lengths = lengths
        self.mask = mask
        self.minlength = minlength
        self.refhash = _vambtools.hash_refnames(identifiers)

    @property
    def nseqs(self) -> int:
        "Number of sequences after filtering"
        return len(self.identifiers)

    def filter_mask(self, mask: Sequence[bool]):
        "Filter contigs given a mask whose length should be nseqs"
        assert len(mask) == self.nseqs
        ind = 0
        for i in range(len(self.mask)):
            if self.mask[i]:
                self.mask[i] &= mask[ind]
                ind += 1

        self.identifiers = self.identifiers[mask]
        self.lengths = self.lengths[mask]
        self.refhash = _vambtools.hash_refnames(self.identifiers)

    def filter_min_length(self, length: int):
        "Set or reset minlength of this object"
        if length <= self.minlength:
            return None

        self.filter_mask(self.lengths >= length)
        self.minlength = length


C = TypeVar("C", bound="Composition")


class Composition:
    """A class containing a CompositionMetaData and its TNF matrix.
    Current fields are:
    * metadata: A CompositionMetaData object
    * matrix: The composition matrix itself
    """

    __slots__ = ["metadata", "matrix"]

    def __init__(self, metadata: CompositionMetaData, matrix: _np.ndarray):
        assert matrix.dtype == _np.float32
        #assert matrix.shape == (metadata.nseqs, 103)

        self.metadata = metadata
        self.matrix = matrix

    def count_bases(self) -> int:
        return self.metadata.lengths.sum()

    @property
    def nseqs(self) -> int:
        return self.metadata.nseqs

    def save(self, io: Union[str, IO[bytes]]):
        _np.savez_compressed(
            io,
            matrix=self.matrix,
            identifiers=self.metadata.identifiers,
            lengths=self.metadata.lengths,
            mask=self.metadata.mask,
            minlength=self.metadata.minlength,
        )

    @classmethod
    def load(cls, io: Union[str, IO[bytes]]):
        if isinstance(io, str):
            arrs = _np.load(io, allow_pickle=True)
            metadata = CompositionMetaData(
                _vambtools.validate_input_array(arrs["identifiers"]),
                _vambtools.validate_input_array(arrs["lengths"]),
                _vambtools.validate_input_array(arrs["mask"]),
                arrs["minlength"].item(),
            )
            return cls(metadata, _vambtools.validate_input_array(arrs["matrix"]))
        
        # Altrimenti, se ci sono più percorsi, li concateniamo
        matrices, identifiers, lengths, masks = [], [], [], []
        item = 0
        old = 0

        for file_path in io:
            data = _np.load(file_path)
            matrices.append(data['matrix'])
            identifiers.append(data['identifiers'])
            lengths.append(data['length'])
            masks.append(data['mask'])
            item = data['minlength']

        # Concateniamo lungo l'asse delle colonne
        concatenated_matrix = _np.concatenate(matrices)
        concatenated_identifiers = _np.concatenate(identifiers)
        concatenated_lengths = _np.concatenate(lengths)
        concatenated_masks = _np.concatenate(masks)
        item = min(old, item)
        old = item
        metadata = CompositionMetaData(
                    _vambtools.validate_input_array(concatenated_identifiers),
                    _vambtools.validate_input_array(concatenated_lengths),
                    _vambtools.validate_input_array(concatenated_masks),
                    item,
                    )

        return cls(metadata, _vambtools.validate_input_array(concatenated_matrix))

    def filter_min_length(self, length: int):
        if length <= self.metadata.minlength:
            return None

        mask = self.metadata.lengths >= length
        self.metadata.filter_mask(mask)
        self.metadata.minlength = length
        _vambtools.numpy_inplace_maskarray(self.matrix, mask)

    @staticmethod
    def _project(fourmers:_np.ndarray, kernel:_np.ndarray = _KERNEL) ->_np.ndarray:
        "Project fourmers down in dimensionality"
        s = fourmers.sum(axis=1).reshape(-1, 1) #sum the content of each row, encolumn the results, the number of rows is the same as before
        s[s == 0] = 1.0
        fourmers *= 1 / s
        fourmers += -(1 / 256)
        return _np.dot(fourmers, kernel)

    @staticmethod
    def _convert(raw: _vambtools.PushArray, projected: _vambtools.PushArray):
        "Move data from raw _vambtools.PushArray to projected _vambtools.PushArray, converting it."
        raw_mat = raw.take().reshape(-1, 256) #I impose only 4^4=256 colmumns, so every row is a single kmer-vector
        projected_mat = Composition._project(raw_mat)
        projected.extend(projected_mat.ravel())
        raw.clear()

    @staticmethod
    def _convert_and_project_mat(raw_mat, kernel=_KERNEL, k=4):
      s = raw_mat.sum(axis=1).reshape(-1, 1)
      s[s == 0] = 1.0
      raw_mat *= 1/s
      raw_mat += -(1/(4**k))  #raw_mat is still with 256 columns
      return _np.dot(raw_mat, kernel)  #dimensions: "-1"x103, every row is a kmer encoded
    
    @staticmethod
    def _pc_project(fourmers:_np.ndarray) ->_np.ndarray:
        s = fourmers.sum(axis=1).reshape(-1, 1) #sum the content of each row, encolumn the results, the number of rows is the same as before
        s[s == 0] = 1.0
        fourmers *= 1 / s
        fourmers += -(1 / 48)
        return fourmers

    @staticmethod
    def _pc_convert_and_project_mat(pc, projected, k=4):
      pc_mat = pc.take().reshape(-1, 48)
      s = pc_mat.sum(axis=1).reshape(-1, 1)
      s[s == 0] = 1.0
      pc_mat *= 1/s
      pc_mat += -(1/(3*2**k))  #raw_mat is still with 48 columns
      projected.extend(pc_mat.ravel())
      pc.clear()

    @staticmethod
    def normalize(fourmers:_np.ndarray) ->_np.ndarray:
        fourmers.reshape(-1, 151)
        abs_sum = _np.abs(fourmers).sum(axis=1).reshape(-1, 1)
        abs_sum[abs_sum == 0] = 1.0
        fourmers *= 1 / abs_sum
        return fourmers

    @classmethod
    def concatenate(cls:type[C], first, second):
        if first == None:
            return second
        concatenated_data = _np.concatenate([first.matrix, second.matrix])
        concatenated_identifiers = _np.concatenate([first.metadata.identifiers, second.metadata.identifiers])
        concatenated_lengths = _np.concatenate([first.metadata.lengths, second.metadata.lengths])
        concatenated_masks = _np.concatenate([first.metadata.mask, second.metadata.mask])
        minlength = min(first.metadata.minlength, second.metadata.minlength)
        metadata = CompositionMetaData(
                    _vambtools.validate_input_array(concatenated_identifiers),
                    _vambtools.validate_input_array(_np.array(concatenated_lengths)),
                    _vambtools.validate_input_array(_np.array(concatenated_masks)),
                    minlength,
                    )
        return cls(metadata, _vambtools.validate_input_array(concatenated_data))

    @classmethod
    def from_file(cls: type[C], filehandle: Iterable[bytes], minlength: int = 100, k=4, use_pc: bool = False, use_tnf: bool = True) -> C:
        """Parses a FASTA file open in binary reading mode, returning Composition.

        Input:
            filehandle: Filehandle open in binary mode of a FASTA file
            minlength: Ignore any references shorter than N bases [100]
        """
        if not use_tnf and not use_pc:
            raise ValueError(f"Both use_pc and use_tnf are False, at least one must be True")
        if minlength < 4:
            raise ValueError(f"Minlength must be at least 4, not {minlength}")

        raw = _vambtools.PushArray(_np.float32)
        pc = _vambtools.PushArray(_np.float32)
        projected_one = _vambtools.PushArray(_np.float32)
        projected_two = _vambtools.PushArray(_np.float32)
        lengths = _vambtools.PushArray(_np.int32)
        mask = bytearray()  # we convert to Numpy at end
        contignames: list[str] = list()

        entries = _vambtools.byte_iterfasta(filehandle)

        for entry in entries: #entry is a single contig taken by the FASTA file
            skip = len(entry) < minlength
            mask.append(not skip)
            if skip:
                continue
            if use_tnf:
                raw.extend(entry.kmercounts(k))
                if len(raw) > 256000:
                    Composition._convert(raw, projected_one)
            if use_pc:
                pc.extend(entry.pcmercounts(k))
                if len(pc) > 48000:
                    Composition._pc_convert_and_project_mat(pc, projected_two)

            lengths.append(len(entry))
            contignames.append(entry.header)
        
        lengths_arr = lengths.take()
        
        metadata = CompositionMetaData(
           _np.array(contignames, dtype=object),
            lengths_arr,
           _np.array(mask, dtype=bool),
            minlength,
        )
        tnfs_arr = _np.empty(shape=(0,), dtype=_np.float32)
        pcs_arr = _np.empty(shape=(0,), dtype=_np.float32)
        # Convert rest of contigs
        if use_tnf:
            Composition._convert(raw, projected_one)
            tnfs_arr = projected_one.take()
            # Don't use reshape since it creates a new array object with shared memory
            tnfs_arr.shape = (len(tnfs_arr) // 103, 103)
            if not use_pc:
                return cls(metadata, tnfs_arr)  #return a new instance of composition, having metadata as data and tnfs_arr as matrix
        if use_pc:
            Composition._pc_convert_and_project_mat(pc, projected_two)
            pcs_arr = projected_two.take()
            pcs_arr.shape = (len(lengths_arr), len(pcs_arr) // len(lengths_arr))
            if not use_tnf:
                return cls(metadata, pcs_arr)  #return a new instance of composition, having metadata as data and pcs_arr as matrix
        return cls(metadata, Composition.normalize(_np.concatenate((tnfs_arr, pcs_arr), axis = 1)))
    @classmethod
    def read_contigs_augmentation(cls: type[C], filehandle, minlength=100, k=4, index_list=None, store_dir="./", backup_iteration=18, augmode=[-1,-1], use_pc = False, use_tnf = True, already = False):
        """Parses a FASTA file open in binary reading mode.

        Input:
            filehandle: Filehandle open in binary mode of a FASTA file
            minlength: Ignore any references shorter than N bases [100]
            backup_iteration: numbers of generation for training
            store_dir: the dir to store the augmentation data
            augmode: the augmentation method. 0 for gaussian noise, 1 for transition, 2 for transversion, 3 for mutation, -1 for all.
            augdatashuffle: whether to shuffle the data from another pool [False, not implemented]

        Outputs:
            tnfs: An (n_FASTA_entries x 103) matrix of tetranucleotide freq.
            contignames: A list of contig headers
            lengths: A Numpy array of contig lengths

        Stores:
            augmentation data (gaussian_noise, transition, transversion, mutation)
        """
        if not use_tnf and not use_pc:
            raise ValueError(f"Both use_pc and use_tnf are False, at least one must be True")
        
        if minlength < 4:
            raise ValueError('Minlength must be at least 4, not {}'.format(minlength))

        norm = _vambtools.PushArray(_np.float32)
        pc = _vambtools.PushArray(_np.float32)
        gaussian = _vambtools.PushArray(_np.float32)
        trans = _vambtools.PushArray(_np.float32)
        traver = _vambtools.PushArray(_np.float32)
        mutated = _vambtools.PushArray(_np.float32)

        lengths = _vambtools.PushArray(_np.int32)
        contignames = list()
        mask = bytearray()
        '''
        # We do not generate the iteration number due to time cost. We just find the minimum augmentation we need for all iteration (backup_iteration)
        # Create backup augmentation pools
        '''
        pool = 2
        gaussian_count, trans_count, traver_count, mutated_count = [0,0], [0,0], [0,0], [0,0]
        # aug_all_method = ['AllAugmentation','GaussianNoise','Transition','Transversion','Mutation']

        # Create projection kernel
        _KERNEL_PROJ = _vambtools.read_npz(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  f"kernel/kernel{k}.npz"))

        # Count the number of entries
        filehandle.seek(0, 0)
        entry_count = _vambtools.count_entry(filehandle)
        print(f'{entry_count} sequences are used for this binning')

        '''If the number of sequences is too large, we might decrease the number of generated augmentation to avoid CLAMB being killed.'''
        if entry_count * backup_iteration > 70000000:
            backup_iteration_2 = 70000000 // entry_count
        else:
            backup_iteration_2 = backup_iteration

        # Pool 1
        for i in range(pool):

            if augmode[i] == -1:
                # Constraits: transition frequency = 2 * transversion frequency = 4 * gaussian noise frequency
                gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = backup_iteration_2 - backup_iteration_2*4//14 - backup_iteration_2*2//14 - backup_iteration_2//2, backup_iteration_2*4//14, backup_iteration_2*2//14, backup_iteration_2//2
            elif augmode[i] == 0:
                gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = backup_iteration_2, 0, 0, 0
            elif augmode[i] == 1:
                gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, backup_iteration_2, 0, 0
            elif augmode[i] == 2:
                gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, backup_iteration_2, 0
            elif augmode[i] == 3:
                gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, 0, backup_iteration_2

            '''Generate augmented data for several batches'''
            pool2 = math.ceil(backup_iteration / backup_iteration_2)
            # Put index for each augmented data
            index = 0
            for i2 in range(pool2):
                backup_iteration_3 = backup_iteration % backup_iteration_2
                if i2 == pool2 - 1 and backup_iteration_3 != 0:
                    # Last batch for generating remaining augmented data
                    if augmode[i] == -1:
                        gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = backup_iteration_3 - backup_iteration_3*4//14 - backup_iteration_3*2//14 - backup_iteration_3//2, backup_iteration_3*4//14, backup_iteration_3*2//14, backup_iteration_3//2
                    elif augmode[i] == 0:
                        gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = backup_iteration_3, 0, 0, 0
                    elif augmode[i] == 1:
                        gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, backup_iteration_3, 0, 0
                    elif augmode[i] == 2:
                        gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, backup_iteration_3, 0
                    elif augmode[i] == 3:
                        gaussian_count[i], trans_count[i], traver_count[i], mutated_count[i] = 0, 0, 0, backup_iteration_3

                '''Reset the file and generator for next reading'''
                filehandle.seek(0, 0)
                entries = _vambtools.byte_iterfasta(filehandle)

                for entry in entries:
                    skip = len(entry) < minlength
                    if i == 0 and i2 == 0:
                        mask.append(not skip)

                    if skip:
                        continue
                    t = _np.empty(0, dtype=_np.int32) 
                    q = _np.empty(0, dtype=_np.int32) 
                    if use_tnf:
                        t = entry.kmercounts(k)
                    if use_pc:
                        q = entry.pcmercounts(k)
                    # t_norm = t / _np.sum(t)
                    # _np.add(t_norm, - 1/(2*4**k), out=t_norm)
                    # print(t_norm)
                    #print(t)
                    if i == 0 and i2 == 0:
                        if use_tnf:
                            norm.extend(t)
                        if use_pc:
                            pc.extend(q)

                    for j in range(gaussian_count[i]):
                        if use_tnf:
                            t_gaussian = mimics.add_noise(t)
                            if not use_pc:
                                gaussian.extend(t_gaussian)
                        if use_pc:
                            q_gaussian = mimics.add_noise(q)
                            if not use_tnf:
                                gaussian.extend(q_gaussian)
                            else:
                                gaussian.extend(_np.concatenate((t_gaussian, q_gaussian)))
                        # print('gaussian',_np.sum(t_gaussian-t_norm))

                    # mutations = mimics.transition(entry.sequence, 1 - 0.021, trans_count[i])
                    if trans_count[i] != 0:
                        mutations = mimics.transition(entry.sequence, 1 - 0.065, trans_count[i])
                    for j in range(trans_count[i]):
                        '''
                        As the function _kmercounts changes the input array at storage, we should reset counts_kmer's storage when using it.
                        '''
                        if use_tnf:
                            counts_kmer = _np.zeros(1 << (2*k), dtype=_np.int32)
                            v._kmercounts(bytearray(mutations[j]), k, counts_kmer)
                        if use_pc:
                            counts_pmer = _np.zeros((3, 1 << k), dtype=_np.int32).reshape(-1)
                            v._pcmercounts(bytearray(mutations[j]), k, counts_pmer)
                            if use_tnf:
                                counts_kmer = _np.concatenate((counts_kmer, counts_pmer))
                            else:
                                counts_kmer = counts_pmer
                        # t_trans = counts_kmer / _np.sum(counts_kmer)
                        # _np.add(t_trans, - 1/(2*4**k), out=t_trans)
                        '''
                        Use deepcopy to avoid storage confusing
                        '''
                        trans.extend(counts_kmer.copy())
                        # print('trans',_np.sum(counts_kmer-t))

                    # mutations = mimics.transversion(entry.sequence, 1 - 0.0105, traver_count[i])
                    if traver_count[i] != 0:
                        mutations = mimics.transversion(entry.sequence, 1 - 0.003, traver_count[i])
                    for j in range(traver_count[i]):
                        if use_tnf:
                            counts_kmer =_np.zeros(1 << (2*k), dtype=_np.int32)
                            v._kmercounts(bytearray(mutations[j]), k, counts_kmer)
                        if use_pc:
                            counts_pmer =_np.zeros((3, 1 << k), dtype=_np.int32).reshape(-1)
                            v._pcmercounts(bytearray(mutations[j]), k, counts_pmer)
                            if use_tnf:
                                counts_kmer = _np.concatenate((counts_kmer, counts_pmer))
                            else:
                                counts_kmer = counts_pmer
                        # t_traver = counts_kmer / _np.sum(counts_kmer)
                        # _np.add(t_traver, - 1/(2*4**k), out=t_traver)
                        traver.extend(counts_kmer.copy())
                        # print('traver',_np.sum(counts_kmer-t))

                    # mutations = mimics.transition_transversion(entry.sequence, 1 - 0.014, 1 - 0.007, mutated_count[i])
                    if mutated_count[i] != 0:
                        mutations = mimics.transition_transversion(entry.sequence, 1 - 0.065, 1 - 0.003, mutated_count[i])
                    for j in range(mutated_count[i]):
                        if use_tnf:
                            counts_kmer =_np.zeros(1 << (2*k), dtype=_np.int32)
                            v._kmercounts(bytearray(mutations[j]), k, counts_kmer)
                        if use_pc:
                            counts_pmer =_np.zeros((3, 1 << k), dtype=_np.int32).reshape(-1)
                            v._pcmercounts(bytearray(mutations[j]), k, counts_pmer)
                            if use_tnf:
                                counts_kmer = _np.concatenate((counts_kmer, counts_pmer))
                            else:
                                counts_kmer = counts_pmer
                        # t_mutated = counts_kmer / _np.sum(counts_kmer)
                        # _np.add(t_mutated, - 1/(2*4**k), out=t_mutated)
                        mutated.extend(counts_kmer.copy())
                        # print('mutated', len(entry), _np.sum(counts_kmer-t))

                    if i == 0 and i2 == 0:
                        lengths.append(len(entry))
                        contignames.append(entry.header)

                # Don't use reshape since it creates a new array object with shared memory
                gaussian_arr = gaussian.take()
                if gaussian_count[i] != 0:
                    if use_tnf and not use_pc:
                        gaussian_arr.shape = (-1, gaussian_count[i], 4**k)
                    elif use_pc and not use_tnf:
                        gaussian_arr.shape = (-1, gaussian_count[i], 3 * 2**k)
                    elif use_tnf and use_pc:
                        gaussian_arr.shape  = (-1, gaussian_count[i], 4**k + 3 * 2**k)
                trans_arr = trans.take()
                if trans_count[i] != 0:
                    if use_tnf and not use_pc:
                        trans_arr.shape = (-1, trans_count[i], 4**k)
                    elif use_pc and not use_tnf:
                        trans_arr.shape = (-1, trans_count[i], 3 * 2**k)
                    elif use_tnf and use_pc:
                        trans_arr.shape  = (-1, trans_count[i], 4**k + 3 * 2**k)
                traver_arr = traver.take()
                if traver_count[i] != 0:
                    if use_tnf and not use_pc:
                        traver_arr.shape = (-1, traver_count[i], 4**k)
                    elif use_pc and not use_tnf:
                        traver_arr.shape = (-1, traver_count[i], 3 * 2**k)
                    elif use_tnf and use_pc:
                        traver_arr.shape  = (-1, traver_count[i], 4**k + 3 * 2**k)
                mutated_arr = mutated.take()
                if mutated_count[i] != 0:
                    if use_tnf and not use_pc:
                        mutated_arr.shape = (-1, mutated_count[i], 4**k)
                    elif use_pc and not use_tnf:
                        mutated_arr.shape = (-1, mutated_count[i], 3 * 2**k)
                    elif use_tnf and use_pc:
                        mutated_arr.shape  = (-1, mutated_count[i], 4**k + 3 * 2**k)
                # AllAugmentation','GaussianNoise','Transition','Transversion','Mutation'
                for j2 in range(gaussian_count[i]):
                    gaussian_save = gaussian_arr[:,j2,:]
                    filepath = f"{store_dir+_os.sep}pool{i}_k{k}_index{index_list[i][index]}_GaussianNoise_{j2}.npz"
                    if use_tnf and not use_pc:
                        gaussian_save.shape = (-1, 4**k)
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            new_data = Composition._convert_and_project_mat(gaussian_save, _KERNEL_PROJ, k)
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, Composition._convert_and_project_mat(gaussian_save, _KERNEL_PROJ, k))
                    elif use_pc and not use_tnf:
                        gaussian_save.shape = (-1, 3* 2**k)
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            new_data = Composition._pc_project(gaussian_save)
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, Composition._pc_project(gaussian_save))
                    elif use_tnf and use_pc:
                        gaussian_save.shape = (-1, 4**k + 3* 2**k)
                        cols_first = 4**k
                        cols_second = 3 * 2**k
                        new_data = Composition.normalize(_np.concatenate((Composition._convert_and_project_mat(gaussian_save[:, :cols_first], _KERNEL_PROJ, k), Composition._pc_project(gaussian_save[:, cols_first:(cols_first + cols_second)])), axis = 1))
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, new_data)

                    index += 1
                for j2 in range(trans_count[i]):
                    trans_save = trans_arr[:,j2,:]
                    filepath=f"{store_dir+_os.sep}pool{i}_k{k}_index{index_list[i][index]}_Transition_{j2}.npz"
                    if use_tnf and not use_pc:
                        trans_save.shape = (-1, 4**k)
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            new_data = Composition._convert_and_project_mat(trans_save, _KERNEL_PROJ, k)
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, Composition._convert_and_project_mat(trans_save, _KERNEL_PROJ, k))
                    elif use_pc and not use_tnf:
                        trans_save.shape = (-1, 3* 2**k)
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            new_data = Composition._pc_project(trans_save)
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, Composition._pc_project(trans_save))
                    elif use_tnf and use_pc:
                        trans_save.shape = (-1, 4**k + 3* 2**k)
                        cols_first = 4**k
                        cols_second = 3 * 2**k
                        new_data = Composition.normalize(_np.concatenate((Composition._convert_and_project_mat(trans_save[:, :cols_first], _KERNEL_PROJ, k), Composition._pc_project(trans_save[:, cols_first:(cols_first + cols_second)])), axis = 1))
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, new_data)
                    index += 1

                for j2 in range(traver_count[i]):
                    traver_save = traver_arr[:,j2,:]
                    filepath = f"{store_dir+_os.sep}pool{i}_k{k}_index{index_list[i][index]}_Transversion_{j2}.npz"
                    if use_tnf and not use_pc:
                        traver_save.shape = (-1, 4**k)
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            new_data = Composition._convert_and_project_mat(traver_save, _KERNEL_PROJ, k)
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, Composition._convert_and_project_mat(traver_save, _KERNEL_PROJ, k))
                    elif use_pc and not use_tnf:
                        traver_save.shape = (-1, 3* 2**k)
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            new_data = Composition._pc_project(traver_save)
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, Composition._pc_project(traver_save))
                    elif use_tnf and use_pc:
                        traver_save.shape = (-1, 4**k + 3* 2**k)
                        cols_first = 4**k
                        cols_second = 3 * 2**k
                        new_data = Composition.normalize(_np.concatenate((Composition._convert_and_project_mat(traver_save[:, :cols_first], _KERNEL_PROJ, k), Composition._pc_project(traver_save[:, cols_first:(cols_first + cols_second)])), axis = 1))
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, new_data)
                    index += 1

                for j2 in range(mutated_count[i]):
                    mutated_save = mutated_arr[:,j2,:]
                    filepath=f"{store_dir+_os.sep}pool{i}_k{k}_index{index_list[i][index]}_Mutation_{j2}.npz"
                    if use_tnf and not use_pc:
                        mutated_save.shape = (-1, 4**k)
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            new_data = Composition._convert_and_project_mat(mutated_save, _KERNEL_PROJ, k)
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, Composition._convert_and_project_mat(mutated_save, _KERNEL_PROJ, k))
                    elif use_pc and not use_tnf:
                        mutated_save.shape = (-1, 3* 2**k)
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            new_data = Composition._pc_project(mutated_save)
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, Composition._pc_project(mutated_save))
                    elif use_tnf and use_pc:
                        mutated_save.shape = (-1, 4**k + 3* 2**k)
                        cols_first = 4**k
                        cols_second = 3 * 2**k
                        new_data = Composition.normalize(_np.concatenate((Composition._convert_and_project_mat(mutated_save[:, :cols_first], _KERNEL_PROJ, k), Composition._pc_project(mutated_save[:, cols_first:(cols_first + cols_second)])), axis = 1))
                        if already:
                            existing_data = _np.load(filepath)['arr_0']
                            _np.savez(filepath, _np.concatenate((existing_data, new_data), axis=0))
                        else:
                            _np.savez(filepath, new_data)
                    index += 1

                gaussian.clear()
                trans.clear()
                traver.clear()
                mutated.clear()

                print(time.time(), backup_iteration, backup_iteration_2, backup_iteration_3)

        lengths_arr = lengths.take()
        if use_tnf and not use_pc:
            #Composition._convert(norm, projected)
            norm_arr = norm.take()  #projected.take
            norm_arr.shape = (-1, 4**k)
            norm_arr = Composition._convert_and_project_mat(norm_arr, _KERNEL_PROJ, k)
        
        elif not use_tnf and use_pc:
            norm_arr = pc.take()   #to take pcmer instead of tnfs
            norm_arr.shape = (-1, 3 * 2**k)
            norm_arr = Composition._pc_project(norm_arr)
        
        elif use_tnf and use_pc:
            norm_arr = _np.concatenate((norm.take(), pc.take()), axis = 1)
            cols_first = 4**k
            cols_second = 3 * 2**k
            norm_arr.shape = (-1, 4**k + 3 * 2**k)
            norm_arr = Composition.normalize(_np.concatenate((Composition._convert_and_project_mat(norm_arr[:, :cols_first], _KERNEL_PROJ, k), Composition._pc_project(norm_arr[:, cols_first:(cols_first + cols_second)])), axis = 1))

        metadata = CompositionMetaData(
           _np.array(contignames, dtype=object),
            lengths_arr,
           _np.array(mask, dtype=bool),
            minlength,
        )
        return cls(metadata, norm_arr)
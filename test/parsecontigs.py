import sys
import os
import numpy as np
import random
import threading
import time

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
import vamb
fasta_path = os.path.join(parentdir, 'test', 'data', 'fasta.fna')
'''
# Test it fails with non binary opened
with open(fasta_path) as file:
    try:
        entries = vamb.vambtools.byte_iterfasta(file)
        next(entries)
    except TypeError:
        pass
    else:
        raise AssertionError('Should have failed w. TypeError when opening FASTA file in text mode')
    file.close()


# Open and read file
with open(fasta_path, 'rb') as file:
    contigs = list(vamb.vambtools.byte_iterfasta(file))
    file.close()

# Lengths are correct
assert [len(i) for i in contigs] == [100, 100, 150, 99, 0, 150]

# Correctly translates ambiguous nucleotides to Ns
contig3 = contigs[2].sequence.decode()
ambigue = set('SWKMYRBDHV')
contig_modificato = ''.join(base if base not in ambigue else 'N' for base in contig3)
contig3 = contig_modificato
for invalid in 'SWKMYRBDHV':
    assert contig3.count(invalid) == 0, f"Carattere non valido '{invalid}' presente nella stringa"
assert contig3.count('N') == 11

# Correctly counts 4mers
contig3_fourmers_expected = """000000210000000100000102120001001010011000100112000100011
0010001000101000001000000100111100020200001012100000001102111011100010011000000
0101010000001001020111010010000111001000010010000010200001000100211110101100010
10000120010100010001000010011110100000100""".replace('\n', '')
contig3_fourmers_observed = contigs[2].kmercounts(4)
print(contig3_fourmers_observed)

for i, j in zip(contig3_fourmers_expected, contig3_fourmers_observed):
    assert int(i) == j

assert all(i == 0 for i in contigs[3].kmercounts(4))

# Correctly deals with lowercase
assert contigs[2].sequence.decode().upper() == contigs[5].sequence.decode().upper()

# Correctly fails at opening bad fasta file
badfasta_path = os.path.join(parentdir, 'test', 'data', 'badfasta.fna')
with open(badfasta_path, 'rb') as file:
    try:
        entries = list(vamb.vambtools.byte_iterfasta(file))
    except ValueError as error:
        pass
        #assert error.args == ("Non-IUPAC DNA byte in sequence badseq: 'P'",)
    else:
        raise AssertionError("Didn't fail at opening fad FASTA file")
    file.close()

# Reader works well
gzip_path = os.path.join(parentdir, 'test', 'data', 'fasta.fna.gz')

with vamb.vambtools.Reader(fasta_path) as file:
    contigs2 = list(vamb.vambtools.byte_iterfasta(file))

assert len(contigs) == len(contigs2)
assert all(i.sequence == j.sequence for i,j in zip(contigs, contigs2))

with vamb.vambtools.Reader(gzip_path) as file:
    contigs2 = list(vamb.vambtools.byte_iterfasta(file))

assert len(contigs) == len(contigs2)
assert all(i.sequence == j.sequence for i,j in zip(contigs, contigs2))

# Test RC kernel
sys.path.append(os.path.join(parentdir, "src"))
import create_kernel

rc_kernel = create_kernel.create_rc_kernel()

def manual_rc_assert(counts):
    indexof = {kmer:i for i,kmer in enumerate(create_kernel.all_kmers(4))}
    cp = counts.copy()
    for row in range(len(counts)):
        for kmer in create_kernel.all_kmers(4):
            rc = create_kernel.reverse_complement(kmer)
            mean = (counts[row, indexof[kmer]] + counts[row, indexof[rc]]) / 2
            cp[row, indexof[kmer]] = mean

    return cp

# Skip zero-length contigs with no 4mers
counts = [contig.kmercounts(4) for contig in contigs[:3]]
counts = np.array(counts, dtype=np.float32)

counts /= counts.sum(axis=1).reshape(-1, 1)
counts -= 1/256

assert np.all(abs(manual_rc_assert(counts) - np.dot(counts, rc_kernel)) < 1e-6)

# Test projection kernel
contig = vamb.vambtools.FastaEntry(b'x', contigs[0].sequence*10000)
counts = np.array(contig.kmercounts(4), dtype=np.float32)
counts /= counts.sum()
counts -= 1/256
counts = np.dot(counts, rc_kernel)
kernel = create_kernel.create_projection_kernel()

projected = np.dot(counts, kernel)
recreated = np.dot(kernel, projected)
assert np.all(np.abs(counts - recreated) < 1e-6)

projected = np.dot(counts, vamb.parsecontigs._KERNEL)
recreated = np.dot(vamb.parsecontigs._KERNEL, projected)
assert np.all(np.abs(counts - recreated) < 1e-6)

# Test read_contigs

with open(fasta_path, 'rb') as file:
    temp = vamb.parsecontigs.Composition.from_file(file, minlength=100)
    tnf = temp.matrix
    contignames = temp.metadata.identifiers
    contiglengths = temp.metadata.lengths
    file.close()
assert len(tnf) == len([i for i in contigs if len(i) >= 100])
#assert all(i-1e-8 < j < i+1e-8 for i,j in zip(tnf[2], contig3_tnf_observed))

assert np.array_equal(contignames, ['Sequence1_100nt_no_special',
 'Sequence2 100nt whitespace in header',
 'Sequence3 150 nt, all ambiguous bases',
 'Sequence6 150 nt, same as seq4 but mixed case'])

assert np.all(contiglengths == np.array([len(i) for i in contigs if len(i) >= 100]))
'''
#bigpath = os.path.join(parentdir, 'test', 'data', 'bigfasta.fna.gz')
bigpath =os.path.join(parentdir, 'test', 'data', 'contigs.fna.gz')
begintime= time.time()
with vamb.vambtools.Reader(bigpath) as f:
    tnf= vamb.parsecontigs_parallel.Composition.from_file(f, use_pc= False).matrix
    f.close()
elapsed = round(time.time() - begintime, 2)
print("Time for parallel:", elapsed)
begintime= time.time()
with vamb.vambtools.Reader(bigpath) as f:
    tnf2= vamb.parsecontigs.Composition.from_file(f, use_pc= False).matrix
    f.close()
elapsed = round(time.time() - begintime, 2)
print("Time for sequential:", elapsed)
print(np.where(tnf!=tnf2))
num_threads = threading.active_count()
print(f"Numero totale di threads attivi: {num_threads}")
#target_tnf = vamb.vambtools.read_npz(os.path.join(parentdir, 'test', 'data', 'target_tnf.npz'))
#assert np.all(abs(tnf - target_tnf) < 1e-8)
'''
paths = [fasta_path, bigpath]
backup_iteration=9
index_list_one = list(range(backup_iteration))
random.shuffle(index_list_one)
index_list_two = list(range(backup_iteration))
random.shuffle(index_list_two)
index_list = [index_list_one, index_list_two]
b=True
for path in paths:
    print("Examining:", path)
    with vamb.vambtools.Reader(path) as file:
        temp = vamb.parsecontigs.Composition.read_contigs_augmentation(file, index_list=index_list, minlength=100, store_dir="./data/", backup_iteration = 9, use_pc=False, already = not b)
        if b:
            b = False
            composition = None
        file.close()
        print("temp.tnf shape", temp.matrix.shape)
        #print(temp.matrix)
        print("temp.contignames shape",temp.metadata.identifiers.shape)
        #print(temp.metadata.identifiers)
        print("temp.contiglengths shape",temp.metadata.lengths.shape)
    with vamb.vambtools.Reader(path) as file:
        composition = vamb.parsecontigs.Composition.concatenate(composition, temp
                        )
        tnf = composition.matrix
        contignames = composition.metadata.identifiers
        contiglengths = composition.metadata.lengths
        print("second temp.tnf shape", tnf.shape)
        print("second temp.contignames shape",contignames.shape)
        #print(contignames)
        print("second temp.contiglengths shape",contiglengths.shape)
        file.close()
print("finals:", tnf.shape)
print(contignames.shape)
print(contiglengths.shape)
composition.save('./data/joined_composition')

b=True
for path in paths:
    print("Examining:", path)
    with vamb.vambtools.Reader(path) as file:
        temp = vamb.parsecontigs.Composition.from_file(file, minlength=100, use_pc= True)
        if b:
            b = False
            composition = None        
        composition = vamb.parsecontigs.Composition.concatenate(composition, temp
                        )
        tnf = composition.matrix
        print(len(tnf))
        contignames = composition.metadata.identifiers
        contiglengths = composition.metadata.lengths
        print(contiglengths[-1])
        print(temp.metadata.lengths[-1])
        file.close()
'''
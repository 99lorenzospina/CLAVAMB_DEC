#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

cpdef int _overwrite_matrix(float[:,::1] matrix, unsigned char[::1] mask):
    """Given a float32 matrix and Uint8 mask, does the same as setting the first
    rows of matrix to matrix[mask], but in-place.
    This is only important to save on memory.
    """

    cdef int i = 0
    cdef int j = 0
    cdef int matrixindex = 0
    cdef int length = matrix.shape[1]
    cdef int masklength = len(mask)

    # First skip to the first zero in the mask, since the matrix at smaller
    # indices than this should remain untouched.
    for i in range(masklength):
        if mask[i] == 0:
            break

    # If the mask is all true, don't touch array.
    if i == masklength:
        return masklength

    matrixindex = i

    for i in range(matrixindex, masklength):
        if mask[i] == 1:
            for j in range(length):
                matrix[matrixindex, j] = matrix[i, j]
            matrixindex += 1

    return matrixindex

cpdef void _kmercounts(unsigned char[::1] bytesarray, int k, int[::1] counts):
    """Count tetranucleotides of contig and put them in counts vector.

    The bytearray is expected to be np.uint8 of bytevalues of the contig.
    Only values 65, 67, 71, 84 are accepted, all others are skipped.
    The counts is expected to be an array of 4^k 32-bit integers with value 0.
    """

    cdef unsigned int kmer = 0
    cdef int character, charvalue, i
    cdef int countdown = k-1
    cdef int contiglength = len(bytesarray)
    cdef int mask = (1 << (2 * k)) - 1
    cdef unsigned int* lut = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    for i in range(contiglength):
        character = bytesarray[i]
        charvalue = lut[character]

        if charvalue == 4:
            countdown = k

        kmer = ((kmer << 2) | charvalue) & mask

        if countdown == 0:
            counts[kmer] += 1
        else:
            countdown -= 1

cpdef void _pcmercounts(unsigned char[::1] bytesarray, int k, int[::1] counts):
    """Compute of contig and put them in counts vector.

    The bytearray is expected to be np.uint8 of bytevalues of the contig.
    Only values 65, 67, 71, 84 are accepted, all others are skipped.
    The counts is expected to be an array of 3*2^k 32-bit integers with value 0.
    A is 0, C is 1, G is 2, T and U are 3
    """

    cdef int character, charvalue, i
    cdef int contiglength = len(bytesarray)
    cdef unsigned char* lut = [
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, #15
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, #31
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, #47
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, #63
        4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, #79
        4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, #95
        4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, #111
        4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, #127
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ]

    cdef double prev = 0.5
    cdef int current = 0
    cdef int cgr = 0
    cdef unsigned int maxvalue = (1<<k)

    for i in range(contiglength):
        character = bytesarray[i]
        charvalue = lut[character]

        if charvalue == 1 or charvalue == 3:
            cgr=maxvalue
        if charvalue == 0 or charvalue == 2:
            cgr=0
        prev=0.5*(prev+cgr)
        current=int(prev)
        prev=current
        counts[current]=counts[current]+1

    prev = 0.5

    for i in range(contiglength):
        character = bytesarray[i]
        charvalue = lut[character]

        if charvalue == 1 or charvalue == 0:
            cgr=maxvalue
        if charvalue == 3 or charvalue == 2:
            cgr=0
        prev=0.5*(prev+cgr)
        current=int(prev)
        prev=current
        counts[current+maxvalue]=counts[current+maxvalue]+1

    prev = 0.5

    for i in range(contiglength):
        character = bytesarray[i]
        charvalue = lut[character]

        if charvalue == 1 or charvalue == 2:
            cgr=maxvalue
        if charvalue == 0 or charvalue == 3:
            cgr=0
        prev=0.5*(prev+cgr)
        current=int(prev)
        prev=current
        counts[current+(maxvalue<<1)]=counts[current+(maxvalue<<1)]+1

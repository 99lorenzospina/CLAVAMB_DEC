"""CL-AVAMB
Documentation: https://github.com/99lorenzospina/CLAVAMB_DEC/tree/avamb_new/vamb

Vamb contains the following modules:
vamb.vambtools
vamb.parsecontigs
vamb.parsebam
vamb.encode
vamb.cluster
vamb.benchmark
vamb.mimic
vamb.aamb_encode

General workflow:

2) Map reads to contigs to obtain BAM files (using minimap2)
3) Calculate a Composition of contigs using vamb.parsecontigs
4) Create Abundance object from BAM files using vamb.parsebam
5) Train autoencoder using vamb.encode
6) Cluster latent representation using vamb.cluster
7) Split bins using vamb.vambtools

Based on AVAMB v4 of Jakob Nybo Nissen and Simon Rasmussen, https://github.com/RasmussenLab/vamb
and CLMB v1 of Pengfei Zhang, https://github.com/zpf0117b/CLMB
"""

__authors__ = "Lorenzo Spina"
__licence__ = "MIT"
__version__ = (1, 0, 0, "DEV")

import sys as _sys

if _sys.version_info[:2] < (3, 9):
    raise ImportError("Python version must be >= 3.9")

from . import vambtools
from . import parsebam
from . import parsecontigs
from . import cluster
from . import benchmark
from . import encode
from . import mimics
from . import species_number
from . import aamb_encode

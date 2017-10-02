# cell_decoder/__init__.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under a BSD "two-clause" license.
# ======================================================================
'''
Cell DECODER: Cell DEep learning and COmputational DEscriptor toolbox.

Python modules and trained neural networks for cell biologists,
implemented in CNTK 2.1+.

Images and metadata from The Human Protein Atlas, where applicable, were used
under a CC BY-NC-ND 4.0 License. See 'human_protein_atlas_license.txt' in
the root folder.

Some functions were adapted from the Microsoft CNTK examples, under an MIT
license.

Copyright (c) 2017 Jeffrey J. Nirschl
MIT license
'''

# cell_decoder version
__version__ = '0.1.0'


# Imports
import imp

# Check whether cntk is installed
try:
    imp.find_module('cntk')
    found = True
except ImportError:
    raise ImportError('Cell DECODER requires the CNTK package.\n',
                      'Please follow the CNTK installation link',
                      'on the home page before continuing.')

#
from . import data
from . import extract
from . import img
from . import io
from . import models
from . import readers
from . import utils
from . import visualize

#print('Running CNTK version {0:s}'.format(cntk.__version__))
#print('Running Cell DECODER version {0:s}'.format(__version__))

##
def __human_protein_atlas_license__():
    hpa_license = './human_protein_atlas_license.txt'
    with open(hpa_license,'r') as lic_file:
        for line in lic_file:
            print(line, end='')
    
def __cell_decoder_license__():
    license_file = './cell_decoder_license.txt'
    with open(license_file,'r') as lic_file:
        for line in lic_file:
            print(line, end='')

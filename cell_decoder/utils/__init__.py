#  cell_decoder/utils/__init__.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ======================================================================
'''

.. module:cell_decoder.utils
    :synopsis: General utilities

'''

# Imports
import os
import glob
import re
import numpy as np

##    
def string_cmp(str1,
               str2,
               match_case=False):
    '''
    string_cmp(str1, str2, match_case=False)
    
    Compare two strings (str) and return a BOOL array
    with the indices where str1 matches str2.

    Default is not case sensitive (match_case=False).
    '''
    assert isinstance(str1, str), "Str1 must be a string!"
    assert isinstance(str2, str), "Str2 must be a string!"
    
    if match_case:
        match_idx = [idx for idx, elem in enumerate(str2) if elem in str1]
        diff_idx = [idx for idx, elem in enumerate(str2) if elem not in str1]
    else:
        match_idx = [idx for idx, elem in enumerate(str2) if elem.lower() in str1.lower()]
        diff_idx = [idx for idx, elem in enumerate(str2) if elem.lower() not in str1.lower()]
    
    return match_idx, diff_idx

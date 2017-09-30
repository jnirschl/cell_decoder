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
Cell DECODER general utilities

Copyright (c) 2017 Jeffrey J. Nirschl
'''

# Imports
import os
import glob
import re
import numpy as np

from cntk.logging.graph import find_by_name, get_node_outputs

##
def print_model(trained_model):
    '''
    utils.print_model(trained_model)
    '''
    # TODO assert model class
    
    # Get node outputs
    node_outputs = get_node_outputs(trained_model)

    # Print
    for l in node_outputs:
        print("  {0} {1}".format(l.name, l.shape))


##    
def string_cmp(str1, str2, match_case=False):
    ''' Compare two strings and return a BOOL array
    with the indices where str1 matches str2.

    Default is not case sensitive (match_case=False).
    
    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
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

##
def recent(model_save_dir, filter_spec='*.dnn*'):
    '''find_recent_model(model_save_dir, load_model=True)

    Accepts a directory (str) and returns a filepath (str).
    '''
    assert os.path.isdir(model_save_dir), \
        ('Model must be a valid directory!')
    
    saved_models = glob.glob(os.path.normpath(os.path.join(model_save_dir, filter_spec))) 
    saved_models = [os.path.basename(elem) for elem in saved_models]
    model_num = []
    for elem in saved_models:
#        print(elem)
        model_num.append(int(re.split('_', elem)[-1].split(".")[0]))
        
    full_path = os.path.normpath(os.path.join(model_save_dir, saved_models[np.argmax(model_num)]))
    model_path = os.path.split(full_path)[0]
    model_name = os.path.split(full_path)[1]
    
    if os.path.isfile(os.path.join(model_path, model_name)):
        print("Found model {}".format(model_name))
    else:
        print("No models found in directory:\n{}".format(model_save_dir))
        
    return model_path, model_name

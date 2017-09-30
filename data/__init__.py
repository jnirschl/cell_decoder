# cell_decoder/data/__init__.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ======================================================================
'''
Cell DECODER function to load sample datasets.

Copyright (c) 2017 Jeffrey J. Nirschl
'''

# Imports
import os
import pandas as pd

# cell_decoder imports
from cell_decoder.visualize import plot


##
def load(dataset):
    '''
    load
    
    Returns a Holoviews RGB element (images) or a 
    Pandas dataframe (datasets).
    '''
    assert isinstance(dataset,str), \
        TypeError('Dataset must be specified as a string')
    VALID_DATASETS = ['hpa_cell_images', 'hpa_structure_images',
                      'hpa_cell_features','hpa_structure_features',
                      'cell_line_prevalence']
    if dataset not in VALID_DATASETS:
        raise ValueError('Invalid dataset name ''{0:s}'.format)


    # Set root dir
    root_dir = os.path.__file__
    
    # Load data
    if dataset == 'hpa_cell_images':
        mapfile_root = os.path.dirname('../../mapfiles/human_protein_atlas/cell_lines/')
        mapfile_name = 'all_cells_mapfile.tsv'
        mapfile_filepath = os.path.join()
        hv_img, df = plot.unique_images(mapfile,
                                       text_labels=text_labels,
                                       randomize=True,
                                       unique=True,
                                       save_im=True,
                                       max_size=224)
    elif 'hpa_structure_images':
        mapfile = 1
        hv_img, df = plot.unique_images(mapfile,
                                       text_labels=self.text_labels,
                                       randomize=True,
                                       unique=True,
                                       save_im=True,
                                       max_size=224)
    elif 'hpa_cell_features':
        filename = '20170921_gscholar_cell_line_prevalence.tsv'
        df = pd.read_csv(os.path.join(root_dir, filename))
    elif 'hpa_structure_features':
        filename = 'Res50_81_cells_mapped_labels.csv'
        df = pd.read_csv(os.path.join(root_dir, filename))
    elif 'cell_line_prevalence':
        filename = '20170921_gscholar_cell_line_prevalence.tsv'
        df = pd.read_csv(os.path.join(root_dir, filename))

    # Return depends on var dataset
    if dataset in VALID_DATASETS[0:2]:
        return df, hv_img
    else:
        return df
        
    

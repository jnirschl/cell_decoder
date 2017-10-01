# Cell_decoder/data/__init__.py
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
import cell_decoder as c_decoder
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
    VALID_DATASETS = ['hpa_cell_mapfile', 'hpa_structure_mapfile',
                      'hpa_cell_features','hpa_structure_features',
                      'cell_line_prevalence']
    if dataset not in VALID_DATASETS:
        raise ValueError('Invalid dataset name ''{0:s}'.format(dataset))

    # Set global root dir
    root_dir = os.path.dirname(c_decoder.__file__)

    # Load data
    if dataset == 'hpa_cell_mapfile':
        mapfile_root = os.path.normpath(os.path.join(root_dir,'mapfiles',
                                                     'human_protein_atlas',
                                                     'cell_lines'))
        mapfile_name = 'all_cells_mapfile.tsv'
        mapfile_filepath = os.path.join(mapfile_root, mapfile_name)
        hv_img, df = plot.unique_images(mapfile_filepath,
                                        text_labels=None,
                                        randomize=True,
                                        unique=True,
                                        save_im=True,
                                        max_size=224)
    elif 'hpa_structure_mapfile':
        mapfile_root = os.path.normpath(os.path.join(root_dir,'mapfiles',
                                                     'human_protein_atlas',
                                                     'cell_lines'))
        mapfile_name = 'all_struct_mapfile.tsv' #TODO add structure mapfile
        mapfile_filepath = os.path.join(mapfile_root, mapfile_name)
        hv_img, df = plot.unique_images(mapfile,
                                        text_labels=text_labels,
                                        randomize=True,
                                        unique=True,
                                        save_im=True,
                                        max_size=224)
    elif 'hpa_cell_features':
        file_root = os.path.normpath(os.path.join(root_dir,'data',
                                                  'cell_lines'))
        filename = 'Res50_81_cells_featvec.csv'
        filepath = os.path.join(root_dir, filename)
        df = pd.read_csv(filepath, sep='\t')
    elif 'hpa_structure_features':
        file_root = os.path.normpath(os.path.join(root_dir,'data',
                                                  'profiling',
                                                  'cells'))
        filename = 'Res50_81_cells_mapped_labels.csv'  #TODO add structure features
        filepath = os.path.join(root_dir, filename)
        df = pd.read_csv(filepath)
    elif 'cell_line_prevalence':
        filename = '20170921_gscholar_cell_line_prevalence.tsv'
        df = pd.read_csv(os.path.join(root_dir, filename))


    # Return depends on var dataset
    if dataset in VALID_DATASETS[0:2]:
        return df, hv_img
    else:
        return df



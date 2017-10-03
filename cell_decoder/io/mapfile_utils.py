# cell_decoder/io/mapfile.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the laboratory of Erika Holzbaur at
# the University of Pennsylvania.
#
# Distributable under an MIT License
# ======================================================================
'''

.. module:cell_decoder.io.mapfile_utils
    :synopsis: Utilities for reading and editing mapfile.

'''

# Standard library imports
from __future__ import print_function

import glob
import os
import re
import time
import numpy as np
import pandas as pd

# cell_decoder imports
import cell_decoder as c_decoder
from cell_decoder.utils import ml_utils


# Set root dir
root_dir = os.path.join(os.path.dirname(c_decoder.__file__), 'mapfiles')

## Read a mapfile and check whether a subset of images exist
def read(mapfile,
         frac_sample=None,
         text_labels=None):
    '''
    df, mapfile_root, mapfile_name = read(mapfile)

    Accepts a STR with the full path to a mapfile
    and returns a Pandas DataFrame, the mapfile directory,
    and the mapfile filename.
    '''
    # Check for errors
    mapfile_root = os.path.dirname(mapfile)
    mapfile_name = os.path.basename(mapfile)

    assert os.path.isdir(mapfile_root), \
        "Mapfile root is not a valid directory!\n\t{0:s}".format(mapfile_root)
    assert os.path.isfile(mapfile), \
        "Mapfile does not exist!\n\t{0:s}".format(mapfile)

    # Save file_ext
    file_ext = os.path.splitext(mapfile_name)[1]

    # Read mapfile
    if file_ext =='.tsv':
        df = pd.read_csv(mapfile, sep='\t', header=None)
    elif file_ext =='.csv':
        df = pd.read_csv(mapfile, sep=',', header=None)
    else:
        raise TypeError('Unrecognized mapfile extension ({0:s)}!'.format(file_ext))

    # Check whether a random sample of the image files exit
    if frac_sample:
        check_images(df=df, frac_sample=frac_sample)

    # Rename the column headers
    if np.greater(df.shape[1], 1):
        df = df.rename(columns={0:'filepath', 1:'label'})
    else:
        df = df.rename(columns={0:'filepath'})

    #TODO merge df with text labels, if available
    # N unique elements in df['label'] and text_df['class'] must
    # be equal
    if text_labels:
        assert (os.path.isfile(text_labels)), \
            "File with text labels does not exist!\n\t{0:s}".format(mapfile)

    # Labels must be indexed from zero CNTK uses zero indexing
    if df['label'].min() !=0:
        raise ValueError('CNTK uses zero indexing for class labels!\n' +
                         'Please check mapfile:\t{0:s}.'.format(mapfile_name))

    # Set the number of unique classes
    unique_labels = np.sort(df['label'].unique())
    if np.diff(unique_labels).max() > 1:
        # Find the missing classes
        missing_labels =  np.setxor1d(unique_labels,
                                      np.arange(0,unique_labels.max()+1))
        print_text = 'One or more classes have no examples:\t{0:s}'
        print(print_text.format(np.array2string(missing_labels)))

    return df, mapfile_root, mapfile_name


##
def save(mapfile=None,
         df=None,
#         save_filename,
         savepath=root_dir,
         sep='\t',
         header=False,
         index=False):
    '''
    save()

    '''
    assert os.path.isdir(savepath), 'Invalid save path!'.format(savepath)
    assert (isinstance(df, pd.DataFrame) or (isinstance(df,list)) and True), \
        'A Pandas Dataframe or a list of DataFrames is required!'

    # Convert to list
    if isinstance(df, pd.DataFrame):
        df = [df]
        save_filename = [save_filename]

    assert isinstance(save_filename, (str)), 'Filename must be a valid string!'

    ## TODO - update save function
    if True:
        raise RuntimeError('Function not complete!')


    # Save all dataframes in the list
    for sub_df, temp_save_name in zip(df, save_filename):
        temp_filepath = os.path.join(savepath, temp_save_name)

        if os.path.isfile(temp_filepath):
            raise ValueError('File {0:s} already exists!'.format(temp_save_name))

        sub_df.to_csv(temp_filepath,
                      sep=sep, header=header, index=index)


## TODO Create mapfile from a folder of images
def create(input_dir,
           label_map,
           filter_spec='png',
           savepath=root_dir,
           save_filename='test.tsv'):
    '''
    create()

    '''
    # Check for errors in input_dir
    if not os.path.isdir(input_dir):
        input_dir = input('Please input a valid directory:')

    assert os.path.isdir(input_dir), \
        "input_dir is not a valid directory!\n\t{0:s}".format(input_dir)

    # Check for errors in label_map
    assert (os.path.isfile(label_map) or \
            isinstance(label_map, (np.ndarray, np.generic))), \
            "label_map must be a valid filepath OR a numpy array"

    if os.path.isfile(label_map):
        label_df = pd.csv_read(label_map, header=False)

    # Get list of files
    file_list = glob.glob(os.path.normpath(os.path.join(input_dir,
                                                        '*' + filter_spec + '*')))

    # Create mapfile dataframe
    df = pd.DataFrame({'filepath':file_list, 'labels':label_map})

    # Save dataframe
    save_mapfile(df, save_filename,
                 savepath=savepath)

    return df

##
def prepend_filepath(mapfile,
                     overwrite=True,
                     prefix=None):
    '''
    df, mapfile = prepend_filepath(mapfile, overwrite=True, prefix=[PATH])

    Accepts a STR with the full path to a mapfile
    and prepends each filepath of the mapfile with
    the specified prefix
    '''
    if isinstance(prefix, str):
        # Read mapfile
        df, mapfile_root, mapfile_name = read(mapfile)

        # Save file_ext
        file_ext = os.path.splitext(mapfile_name)[1]

        # Prepend filepath
        df['filepath'] = prefix + df['filepath'].astype(str)

        if overwrite:
            if file_ext =='.tsv':
                df.to_csv(mapfile, sep='\t', header=None, index=False)
            elif file_ext =='.csv':
                df.to_csv(mapfile, sep=',', header=None, index=False)

    return df, mapfile

##
def append(mapfile):
    '''
    append()

    '''
    #TODO finish function
    mapfile = 1

    return df

##
def check_images(df=None,
                 mapfile=None,
                 n_sample=None,
                 frac_sample=0.01):
    '''
    check_images(mapfile=mapfile, frac_sample=0.01)

    Accepts a Pandas DataFrame or a mapfile and checks
    whether a subset of files in ['filepath'] exist.
    '''
    assert (isinstance(df, pd.DataFrame) \
            or mapfile is not None), \
            'A Pandas Dataframe or mapfile are required!'
    assert (frac_sample > 0 and frac_sample <1), \
        ValueError('Sampling fraction must be' + \
                   ' in the range ({0:0.2f},{1:0.2f})!'.format(0,1))

    # Read mapfile
    if isinstance(df, pd.DataFrame) is None:
        df, _, _ = read(mapfile)

    # Sample a random subset of dataframe, either by n or fraction
    if isinstance(n_sample, str) and n_sample.lower()=='all':
        n_sample = df.shape[0]

    if n_sample:
        df_out = df.sample(n=n_sample, replace=False)
    else:
        df_out = df.sample(frac=frac_sample, replace=False)

    # Adjust for very large mapfiles
    if df_out.shape[0] > 1e3:
        df_out = df.sample(frac=frac_sample/10, replace=False)

    # Check whether images exist
    print_text = 'Checking a random {0:0.2f}% of mapfile to ensure files exist.\n'
    print(print_text.format(frac_sample*100))
    
    for elem in df_out.values:
        assert os.path.isfile(elem[0]), \
            'Image does not exist!\n\t{0:s}'.format(os.path.basename(elem[0]))

    # Read size, bit depth, and num_channels from image header
    # TODO
    # Error check image size and channel number


##
def sample(mapfile=None,
           df=None,
           n_sample=1,
           frac=None,
           grouped=True,
           replace=False):
    '''
    df_supset, df_subset = sample(mapfile=mapfile, n_sample=1)

    Accepts a Pandas DataFrame or filepath to a mapfile and samples
    N elements or a fraction N of elements by group ['label'].

    Returns df_superset (df original less sampling) and df_subset.
    '''
    assert (isinstance(df, pd.DataFrame) or mapfile is not None), \
        ValueError('A Pandas Dataframe or mapfile are required!')

    # Read mapfile
    if not isinstance(df, pd.DataFrame):
        df, mapfile_root, mapfile_name = read(mapfile,
                                              frac_sample=None)

    # Sample the df, group if necessary
    if grouped:
        # Group by label
        df_grouped = df.groupby('label')

        # Sample n or fraction from the whole dataset
        if frac:
            fn = lambda obj: obj.sample(frac=frac, replace=replace)
        else:
            fn = lambda obj: obj.sample(n=np.min((n_sample, obj.shape[0])),
                                        replace=replace)

        df_subset = df_grouped.apply(fn)

    else:
        # Sample n or fraction from the whole dataset
        if frac: #np.greater(n_sample,1):
            # Get a random 10% of the dataframe
            df_subset = df.sample(frac=frac, replace=replace)
        else:
            # Get a random user-specified number from the dataframe
            df_subset = df.sample(n=n_sample, replace=replace)


    # Drop original label column, so we can use reset_index
    df_subset.drop(['label'], axis=1, inplace=True)
    df_subset = df_subset.reset_index()
    df_subset = df_subset.rename(columns={'level_1':'orig_idx'})

    # Get the indices for the dataframe less the subset
    diff_idx_list = np.setdiff1d(df.index.tolist(), df_subset['orig_idx'])
    df_supset = df.ix[diff_idx_list]

    # Drop column orig_idx
    df_subset.drop(['orig_idx'], axis=1, inplace=True)

    return df_supset, df_subset

##
def summarize(df=None,
              mapfile=None,
              group_label='label'):
    '''
    summarize()

    Accepts a Pandas DataFrame or filepath to a mapfile and
    summarizes the count and percentage of each class in the
    dataset.
    '''
    assert (isinstance(df, pd.DataFrame) or mapfile is not None), \
        'A Pandas Dataframe or valid filepath to a mapfile are required!'

    # Read mapfile, if necessary
    if df is None:
        group_label = 'label'
        df, _, _ = read(mapfile)
        #    else:
        # TODO - check pandas label
        # assert group_label is valid key in the df

    # Summarize the mapfile
    df_count = df.groupby([group_label], as_index=True).count().reset_index()
    df_count = df.groupby(['label'], as_index=True).count().reset_index()
    df_count = df_count.rename(columns={'filepath':'count',
                                        'label':'label'})
    df_percent = pd.DataFrame({'label':df_count['label'],
                               'percent':df_count['count']/df_count['count'].sum()})

    # Merge dfs
    df_out = df_count.merge(df_percent, how='left')

    return df_out

##
def crossval(df=None,
             held_out_test=50,
             k_fold=10,
             mapfile=None,
             method='skf',
             random_seed=None,
             save=True,
             save_root=root_dir):
    '''
    df_train, df_held_out = crossval()

    Accepts a Pandas DataFrame or filepath to a mapfile and
    partitions the dataset into training, with cross validation,
    and a held-out dataset. Mapfiles are saved to the mapfile
    directory.

    Returns a df_train (training + cross-validation) and df_held_out.
    '''
    
    # Read mapfile or get df
    if mapfile is None:
        assert isinstance(df, pd.DataFrame), \
            'df must be a Pandas DataFrame!'
    elif df is None:
        assert (isinstance(mapfile, str) and os.path.isfile(mapfile)), \
            'Mapfile must be a valid file!'
        df, mapfile_root, _ = read(mapfile)
        save_root = mapfile_root
    else:
        raise ValueError('A valid mapfile or Pandas dataframe is required!')      

    # Separate a held-out test set
    if held_out_test > 0 and held_out_test < 1:
        # Assume held_out_test is a fraction.
        # Set aside a fraction of each class as a held-out test
        df, df_held_out = sample(df=df,
                                 n_sample=None,
                                 frac=held_out_test,
                                 grouped=True,
                                 replace=False)

    elif held_out_test > 0:
        # Assume literal N samples to hold out per class
        # Set aside a fixed number of each class as a held-out test
        df_summary = summarize(df=df)

        # Adjust held-out sampling, given min class size
        if df_summary['count'].min() <= held_out_test:
            held_out_test = np.floor(df_summary['count'].min()/2).astype(int)

        # Sample df by group
        df, df_held_out = sample(df=df,
                                 n_sample=held_out_test,
                                 frac=None,
                                 grouped=True,
                                 replace=False)

        # Adjust k_fold based on min class number
        df_summary = summarize(df=df)
        
        if df_summary['count'].min() < k_fold:
            min_idx = np.argmin(df_summary['count'])
            k_fold = df_summary['count'].min()
            
            print_text = 'Class {0:d} has too few examples ({1:d})' + \
                         'for {2:d} fold cross-validation!\nUsing {1:d}' +\
                         ' fold cross-validation instead!'
            print(print_text.format(df_summary['label'].ix[min_idx],
                                    df_summary['count'].ix[min_idx],
                                    k_fold))
            
    else:
        # No held-out test
        df_held_out = None

    # Set up cross validation
    df = ml_utils.cv_partition(df=df,
                               data='filepath',
                               label='label',
                               method=method,
                               k_fold=k_fold,
                               random_state=random_state)

    # Save output
    if save:
        print('Saving files to\n\t{0:s}'.format(save_root))

        # Save mapfiles
        save_filename = '{0:02d}_train_mapfile.tsv'
        save_filepath = os.path.join(save_root, save_filename)    
        train_mapfile_list = [elem.format(fold) for fold, elem in zip(range(0, k_fold)), [save_filepath]*k_fold]
        test_mapfile_list = [elem.replace('train', 'test') for elem in train_mapfile_list]

        # 
        for fold, test_df in df.groupby('validation_fold'):
            test_ix = test_df.index
            train_ix = np.setdiff1d(df.index, test_ix)
            
            # Get train df
            train_df = df.ix[train_ix]
            
            # write csv
            train_df.to_csv(save_filepath.format(fold),
                            sep='\t', header=False,
                            index=False)
            test_df.to_csv(save_filepath.replace('train','test').format(fold),
                           sep='\t', header=False,
                           index=False)
        
        # Save train and held-out
        df.to_csv(save_filepath.replace('train','all_train-val'), sep='\t',
                  header=False, index=False)
            
        if df_held_out is not None:
            df_held_out.to_csv(save_filepath.replace('train','held-out-test'),
                               sep='\t', header=False, index=False)

    else:
        train_mapfile_list = None
        test_mapfile_list = None

    return {'train':train_mapfile_list,
            'test':test_mapfile_list}

## TODO coordinate text labels and mapfile labels

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
Cell DECODER mapfile utilities

'''

# Standard library imports
from __future__ import print_function

import glob
import os
import re
import time
import numpy as np
import pandas as pd

# Third-party imports
from sklearn.model_selection import StratifiedKFold

## Read a mapfile and check whether a subset of images exist
def read(mapfile,
         frac_sample=None,
         text_labels=None):
    '''df, mapfile_root, mapfile_name = read(mapfile)

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
        print('One or more classes have no examples:\t{0:s}'.format(np.array2string(missing_labels)))

    return df, mapfile_root, mapfile_name


##
def save(df,
         save_filename,
         savepath='../mapfiles/',
         sep='t',
         header=False,
         index=False):
    '''save

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
           savepath='../mapfiles/',
           save_filename='test.tsv'):
    '''create()
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
def prepend_mapfile_filepath(mapfile,
                             prefix=None):
    '''prepend_mapfile_filepath(mapfile, prefix=[PATH])

    Accepts a STR with the full path to a mapfile
    and prepends each filepath of the mapfile with
    the specified prefix
    '''
    # Read mapfile
    df, mapfile_root, mapfile_name = read(mapfile)

    # Prepend filepath with given prefix
    #    filepaths = df.ix[:,0].values
    #    labels = df.ix[:,1].values
    # Save  mapfile

    return mapfile

##
def append(mapfile):
    '''
    append
    '''
    mapfile = 1

    return df

##
def check_images(df=None,
                 mapfile=None,
                 n_sample=None,
                 frac_sample=0.01):
    '''check_images

    Accepts X and returns Y.
    '''
    assert (isinstance(df, pd.DataFrame) \
            or mapfile is not None), \
            'A Pandas Dataframe or mapfile are required!'
    assert (frac_sample > 0 and frac_sample <1), \
        ValueError('Sampling fraction must be in the range ({0:0.2f},{1:0.2f})!'.format(0,1))

    if isinstance(df, pd.DataFrame) is None:
        # Read mapfile
        df, _, _ = read(mapfile)

    # Sample a random subset of dataframe
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
    print('Checking a random {0:0.2f}% of mapfile to ensure files exist.\n'.format(frac_sample*100))
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
           frac=0.1,
           grouped=True,
           replace=False):
    '''sample

    Accepts X and returns Y.
    '''
    assert (isinstance(df, pd.DataFrame) or mapfile is not None), \
        ValueError('A Pandas Dataframe or valid filepath to a mapfile are required!')

    # 
    if not isinstance(df, pd.DataFrame):
        # Read mapfile - error checking in "read"
        df, mapfile_root, mapfile_name = read(mapfile,
                                              frac_sample=None)

    # Sample the df
    if grouped:
        # Group by label
        df_grouped = df.groupby('label')

        # Sample n per group
        fn = lambda obj: obj.sample(n=np.min((n_sample, obj.shape[0])),
                                    replace=replace)
        df_subset = df_grouped.apply(fn)

    else:
        # Sample n or fraction from the whole dataset
        if np.greater(n_sample,1):
            # Get a random user-specified number from the dataframe
            df_subset = df.sample(n=n_sample, replace=replace)
        else:
            # Get a random 10% of the dataframe
            df_subset = df.sample(frac=frac, replace=replace)

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
def balance_classes(mapfile):
    '''balance_classes

    
    '''
    # Summarize the class count in the mapfile
    df_grouped, _ = summarize(mapfile)

    # Find the minimum count in a single class
    min_n_samples = min(df_grouped['label'].count())

    # Sample min_class n samples from each group
    _, df_subset = sample(mapfile, n_sample=min_n_samples,
                          grouped=True, replace=False)

    return df_subset

##
def summarize(df=None,
              mapfile=None,
              group_label='label'):
    '''summarize

    Accepts X and returns Y.
    '''
    assert (isinstance(df, pd.DataFrame) or mapfile is not None), \
        'A Pandas Dataframe or valid filepath to a mapfile are required!'

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
def crossval(mapfile=None,
             df=None,
             k_fold=10,
             balance_classes=False,
             held_out_test=50,
             save_root=os.getcwd(),
             random_seed=None):
    '''
    crossval

    '''
    if mapfile is None:
        assert isinstance(df, pd.DataFrame), \
            'df must be a Pandas DataFrame!'
    elif df is None:
        assert (isinstance(mapfile, str) and os.path.isfile(mapfile)), \
            'Mapfile must be a valid file!'
        
    # TODO get seed automatically from random.org
    if random_seed is None:
        random_seed = 123456789
        print('Using a fixed random seed of {0:d}'.format(random_seed))

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Read mapfile or get df
    if df is None:
        df, mapfile_root, _ = read(mapfile)
        save_root = mapfile_root

    # Separate a held-out test set
    if held_out_test > 0:
        df_summary = summarize(df=df)

        # Adjust held-out sampling, given min class size
        if df_summary['count'].min() <= held_out_test:
            held_out_test = np.floor(df_summary['count'].min()/2).astype(int)
            
        df, df_held_out = sample(df=df,
                                 n_sample=held_out_test,
                                 frac=None,
                                 grouped=True,
                                 replace=False)

        # Adjust k_fold based on class number
        df_summary = summarize(df=df)
        if df_summary['count'].min() < k_fold:
            min_idx = np.argmin(df_summary['count'])
            print('Class {0:d} has too few examples ({1:d}) for {2:d} fold cross-validation! Using {0:d} fold cross-validation instead'.format(df_summary['label'].ix[min_idx],df_summary['count'].ix[min_idx], k_fold ))
            k_fold = df_summary['count'].min()
            
        #TODO save held out
        
    else:
        df_held_out = None

    # Setup stratified k fold (SKF) cross validation.
    # SKF will maintain the class proportion across
    # all folds (train/ test).
    #TODO update for recent scipy skf
    skf = StratifiedKFold(n_splits=k_fold,
                          shuffle=True,
                          random_state=random_seed)

    all_test_idx = np.zeros((df['label'].shape[0],1), dtype=int)
    for fold, [tr_idx, test_idx] in enumerate(skf.split(df['filepath'],
                                                        df['label'])):
        # Save train and test filepaths and labels in a csv
        save_filepath = os.path.normpath(os.path.join(save_root,
                                                      '{0:d}_all-cells_train_filepath_labels.tsv'.format(fold)))

        # Save test index, all other examples used in training
        all_test_idx[test_idx] = fold

    # Merge test idx with original df
    df = df.join(pd.DataFrame(all_test_idx, columns={'test_fold'}))
        #save_mapfile(df.ix[tr_idx], save_filepath.replace)
        #save_mapfile(df.ix[test_idx], save_filepath.replace('train','test'))

    return df, df_held_out

## TODO coordinate text labels and mapfile labels

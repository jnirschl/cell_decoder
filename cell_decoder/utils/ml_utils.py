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

.. module:cell_decoder.utils.ml_utils
    :synopsis: Machine learning utilities

'''

# Standard library imports
import numpy as np
import pandas as pd


# Third part imports
from sklearn.model_selection import StratifiedKFold

## 
def cv_partition(df=None,
                 data='filepath',
                 label='label',
                 method='skf',
                 k_fold=10,
                 random_seed=None):
    '''
    df_train, df_held_out = cv_partition()

    Accepts a Pandas DataFrame and partitions a trainin dataset into
    k-folds for cross validation.

    Returns a DataFrame with test folds indicated.
    '''
    VALID_METHODS = ['skf', '632bootstrap', 'smote']
    if method not in VALID_METHODS:
        raise ValueError('Invalid method {0:s}!'.format(method))

    # Shuffle the dataframe in place
    df = df.sample(frac=1).reset_index(drop=True) #TODO check
    
    # Process cross validation given method
    if method.lower() == 'skf':
        # Setup stratified k fold (SKF) cross validation.
        skf = StratifiedKFold(n_splits=k_fold,
                              shuffle=True,
                              random_state=random_seed)
               
        # Pre-allocate array
        all_test_idx = np.empty((df[label].shape[0], 1), dtype=int)
        
        for fold, [train_ix, test_ix] in enumerate(skf.split(df[data], df[label])):
            # Assign test index, all other examples used in training
            all_test_idx[test_ix] = fold

    elif method.lower() == '.632bootstrap':
        raise RuntimeError('The {0:s} method is not complete!'.format(method))
    
    elif method.lower() == 'smote':
        raise RuntimeError('The {0:s} method is not complete!'.format(method))

    # Merge test idx with original df
    df = df.join(pd.DataFrame(all_test_idx, columns={'validation_fold'}))

    return df

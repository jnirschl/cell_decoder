#  cell_decoder/io/__init__.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ======================================================================
'''
Cell DECODER feature extraction function

Copyright (c) 2017 Jeffrey J. Nirschl
'''

# Imports
import os

# Array and dataset operations
import numpy as np
import pandas as pd

# Main CNTK imports
from cntk import combine, load_model
from cntk.logging.graph import find_by_name

##
def start(model_path, mapfile, output_file, mb_source=None,
          num_objects=None, node_name='pool5',
          stream_name='features', trained_model=None):
    '''extract_deep_features(model_path, mapfile, output_file, mb_source=None, num_objects=None, node_name='pool5',stream_name='image', trained_model=None)
    
    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    # Load model if None given
    if trained_model == None:
        trained_model  = load_model(model_path)
    
    # Create image reader if None given
    if mb_source == None:
        print('No reader defined.\nCreating new reader from mapfile.')
#        data_struct = defaults.data_struct()
        data_struct = {'label_dim':num_objects,
                       'image_height':224,
                       'image_width':224,
                       'num_channels':3}
        mb_source  = create_reader.image(mapfile, data_struct,
                                         xform_params=None, is_training=False)

    # Set stream_name based on mb_source
    temp_stream_name = mb_source.streams.keys()
    temp_stream_name = [elem for elem in temp_stream_name]
    if temp_stream_name[0]!= stream_name:
        print('Stream name does not match minibatch source! Resetting var stream_name')
        #stream_name = temp_stream_name[0]
    
    # Get the contents of the mapfile
    contents = [[idx, line.split("\t")] for idx, line in enumerate(open(mapfile,'r'))]
    filename_label = [[os.path.split(elem[1][0].strip())[1], int(elem[1][1].strip()) ]  for elem in contents] # 2nd is a tuple of [filepath and label]
    
    # Process all if None given
    if num_objects == None:
        num_objects = 1 + contents[-1][0] # 1st element of contents is an idx, get last elem [-1]
        print("Evaluating all {0} images".format(num_objects))

    node_in_graph = trained_model.find_by_name(node_name)
    output_nodes  = combine([node_in_graph.owner])

    # Define function for saving output csv
    def save_pd_df(feature_vec, filename_label, filename):
        feature_vec_df = pd.DataFrame(feature_vec)
        label_vec_df = pd.DataFrame(filename_label,
                                    columns=['name','label'])

        if os.path.isfile(filename + 'featvec.csv'):
            # append to file
            feature_vec_df.to_csv(filename + '_featvec.csv', mode='a', sep=",",
                                  float_format='%.3f', index=False, header=False)
            label_vec_df.to_csv(filename + '_labels.csv', mode='a', sep=",",
                                index=False, header=False)
        else:
            # write, clear existing contents
            feature_vec_df.to_csv(filename + '_featvec.csv', mode='w', sep=",",
                                  float_format='%.3f', index=False, header=False)
            label_vec_df.to_csv(filename + '_labels.csv', mode='w', sep=",",
                                index=False, header=False)
            
        return feature_vec_df, label_vec_df

    
    # Allocate output variable
    feature_vec = []
    save_filename = os.path.basename(os.path.normpath(output_file)).split('.')[0]
    
    # evaluate model and get desired node output
    print("Evaluating model for output node %s" % node_name)
    features_si = mb_source[stream_name]
    with open(output_file, 'wb') as results_file:
        for i in range(0, num_objects): # num_objects +1 because range() does not include end value
           
            mb = mb_source.next_minibatch(1)
            out_feats = output_nodes.eval(mb[features_si])

            # Flatten list
            out_feats_flat = out_feats[0].flatten()
            
            # Append feature vector and labels to output vars
            feature_vec.append(out_feats_flat)
            
            # Write results to file
            np.savetxt(results_file, out_feats_flat[np.newaxis], fmt="%.5f")

            if (i % 1000) ==0 and (i!=0):
                save_pd_df(feature_vec, filename_label, save_filename)
                print('\tProcessed {0} of {1} images...'.format(i+1,num_objects))
            
    feature_vec_df, label_vec_df = save_pd_df(feature_vec, filename_label, save_filename)
    
    print('\tProcessed {0} of {1} images...'.format(i+1,num_objects))
    return feature_vec_df, label_vec_df
    

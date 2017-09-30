# train_model.py -- a module within cntk_utils
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under a BSD "two-clause" license.
# ==============================================================================
# Adapted from 07_Deconvolution_Visualizer.py from Microsoft.
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk_utils import find_model



def weights(model_path, mapfile):
     '''visualize.weights(map_file_root, train_map, test_map, valid_map)
    
    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    
    return

def deconvolution(data_struct, mb_source=None, num_objects=None,
                  output_node=None,stream_name='features',
                  trained_model=None):
    '''deconvolution()
    
    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    # Load model
    if trained_model is None and data_struct['model'] is None:
        if data_struct['model_path'] is None:
            print('Error! No network.'
        else:
            trained_model  = load_model(data_struct['model_path'])
    else:
        if trained_model is None:
            trained_model = data_struct['model']
    
    # Create  minibatch source
    if mb_source is None:
        if data_struct['reader_test']==None:
                  mb_source  =create_reader.image(data_struct['test_map'],
                                                  data_struct, False)
    else:
        mb_source = data_struct['reader_test']
        
    # Set stream_name based on mb_source
    temp_stream_name = mb_source.streams.keys()
    temp_stream_name = [elem for elem in temp_stream_name]
    if temp_stream_name[0]!= stream_name:
        print('Stream name does not match minibatch source! Resetting var stream_name.')

    # Evaluate model and save output
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
                print('\tProcessed {0} of {1} images...'.format(i+1,num_objects)

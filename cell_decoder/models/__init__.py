# cell_decoder/models/__init__.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ======================================================================
'''
Cell DECODER models

Copyright (c) 2017 Jeffrey J. Nirschl
'''

__all__ = ['AlexNet',
           'faster_rcnn',
           'GoogLeNet_BN_inception',
           'GoogLeNet_inceptionV3',
           'resnet_blocks',
           'resnet_deep',
           'resnet_shallow',
           'vgg16',
           'vgg19']

# Imports
import os
import copy

import numpy as np
import pandas as pd
import pickle
from scipy.stats import kstest

# Main CNTK imports
import cntk as C
from cntk import combine, load_model, softmax, CloneMethod, Trainer, UnitType
from cntk.debugging import enable_profiler,start_profiler, stop_profiler
from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed
from cntk.metrics import classification_error

# cell_decoder inputs
from cell_decoder.config import DataStructParameters, LearningParameters, ResNetParameters, TransformParameters
from cell_decoder.io import mapfile_utils
from cell_decoder.models import resnet_deep, resnet_shallow
from cell_decoder.readers import default_reader, microscopy_reader
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.progress_print import TensorBoardProgressWriter
from cntk.losses import cross_entropy_with_softmax


##
def create(model_parameters,
           num_classes=None,
           pixel_mean=25,
           resnet_layers=50,
           image_height=224,
           image_width=224,
           num_channels=3,
           scaling_factor=0.00390625,
           num_stack_layers=[2,3,5,2],
           use_mean_image=False):
    '''
    models.create()

    Returns a dictionary "model_dict" with the keys input_var, label_var, and net.
    '''
    # Get ModelParameters
    if model_parameters:
        # Check model keys
        valid_model_params = ResNetParameters()

        for elem in model_parameters.__dict__:
            if elem not in valid_model_params.__dict__.keys():
                raise TypeError('Invalid model parameter')

    else:
        model_parameters = ResNetParameters(bias=bias,
                                            bn_time_const=bn_time_const,
                                            dropout=dropout,
                                            init=init,
                                            num_classes=num_classes,
                                            num_filters=num_filters,
                                            pad=pad,
                                            pixel_mean=pixel_mean,
                                            resnet_layers=resnet_layers,
                                            num_stack_layers=num_stack_layers,
                                            use_mean_image=use_mean_image)

    #
    if num_classes is None:
        num_classes = len(df['labels'].unique())
        
    # Assign vars from model_params
    bias = model_parameters.bias
    bn_time_const = model_parameters.bn_time_const
    dropout = model_parameters.dropout
    init = model_parameters.init
    model_name = model_parameters.model_name
    model_path = model_parameters.model_path
    network_name = model_parameters.network_name
    num_filters = model_parameters.num_filters
    num_stack_layers = model_parameters.num_stack_layers
    pad = model_parameters.pad
    resnet_layers = model_parameters.resnet_layers

    # Set input
    input_var = C.input_variable((num_channels,
                                  image_height,
                                  image_width),
                                 name='input')

    # The pixel mean is subtracted in the model. The mean image
    # is subtracted while creating the mb_source (e.g. in ImageDeserializer)
    if use_mean_image:
        input_var = C.ops.minus(input_scaled,
                                C.ops.constant(pixel_mean),
                                name="input_mean_sub")

    # Scale 8 bit image to 0-1
    input_var = C.ops.element_times(input_var,
                                    C.ops.constant(scaling_factor),
                                    name="input_mean_sub_scaled")

    # Set num_classes
    label_var = C.input_variable( num_classes )

    # Create resnet stack
    if resnet_layers in [18, 34]:
        ##TODO - not complete
        net = resnet_shallow(input_var, num_stack_layers,
                             num_filters, num_classes,
                             bias, bn_time_const,
                             dropout, init, pad)

    elif resnet_layers in [50, 101, 152]:
        net = resnet_deep.bottleneck(input_var, num_stack_layers,
                                     num_filters, num_classes,
                                     bias, bn_time_const, dropout, init,
                                     pad)

    # Assign model dictionary
    model_dict = {'input_var':input_var,
                  'label_var':label_var,
                  'net':net,
                  'num_classes':num_classes}

    return model_dict

##
def train(model_dict, # input_var', 'label_var', 'net]'
          reader_train,
          train_epoch_size,
          learn_params,
          num_classes,
          debug_mode=False,
          gpu=True,
          model_save_root=None,
          profiler_dir=None,
          reader_test=None,
          tb_log_dir='C:/TensorBoard_logs/cntk',
          tb_freq=10,
          test_epoch_size=None,
          extra_aug=True):
    '''models.train()

    Returns the trained network and a history of the training accuracy/ loss and test accuracy.
    '''
    # Error check
    valid_model_dict = ['input_var', 'label_var',
                        'net', 'num_classes']
    for elem in model_dict.keys():
        if elem not in valid_model_dict:
            raise TypeError('Invalid model dictionary')

    # Set device
    if gpu:
        C.device.try_set_default_device(C.device.gpu(0))

    # Set input var
    input_var = model_dict['input_var']
    label_var = model_dict['input_var']

    # Set network
    net = model_dict['net']

    # Set debug opts
    if debug_mode:
        print('Debug mode enabled.\n')
        train_epoch_size = 1
        set_computation_network_trace_level(0)
        set_fixed_random_seed(260732) # random number from random.org

    # Set the learning  parameters
    mb_size = learn_params['mb_size']
    momentum_time_constant = learn_params['momentum_time_const']
    lr_per_sample = learn_params['lr_per_mb']
    lr_schedule = learn_params['lr_schedule']
    mm_schedule = learn_params['mm_schedule']

    # Define classification loss and eval metrics
    loss_fn = cross_entropy_with_softmax(net, label_var)
    eval_fn = classification_error(net, label_var)
#        evaluationNodes = (errs) # top5Errs only used in Eval


    # progress writers
    progress_printer = [ProgressPrinter(tag='Training',
                                        num_epochs=learn_params['max_epochs'])]

    # Setup TensorBoard logging
    if tb_log_dir and isinstance(tb_log_dir, str):
        try: # Check the save directory
            os.stat(tb_log_dir)
        except:
            os.mkdir(tb_log_dir)

        tensorboard_writer = TensorBoardProgressWriter(freq=tb_freq,
                                                       log_dir=tb_log_dir,
                                                       model=net)
        progress_printer.append(tensorboard_writer)
    else:
        tensorboard_writer = None

    # Setup model_save_root
    if model_save_root and isinstance(model_save_root, str):
        try:
            os.stat(model_save_root)
        except:
            os.mkdir(model_save_root)

    # trainer object
    if learn_params['learner']=='adadelta':
        raise RuntimeError('Not yet implemented')
#        print('Training using adadelta')
#        learner = adadelta(net.parameters, lr_schedule)
    elif learn_params['learner']=='sgd':
        print('Training using momentum_sgd')
        learner = momentum_sgd(net.parameters,
                               lr_schedule,
                               mm_schedule,
                               l2_regularization_weight=learn_params['l2_reg_weight'])

    # Create training obj
    trainer = Trainer(net, (loss_fn, eval_fn), learner, progress_printer)

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    # Print the number of parameters
    log_number_of_parameters(net) ; print()

    # TODO - Save data_struct dictionary as pickle
#    save_filename = os.path.join(data_struct['model_save_root'],
#                                 data_struct['network_name'] + "_data-struct.pickle")
#    with open(save_filename, 'wb') as save_handle:
        #pickle.dump(data_struct, save_handle, protocol=pickle.HIGHEST_PROTOCOL)


    # perform model training
    if profiler_dir:
        start_profiler(profiler_dir, True)

    # Train over (1, max_epochs)
    cumulative_count = 0
    for epoch in range(learn_params['max_epochs']):
        sample_count = 0 #   running_loss = []

        # Train over minibatches in epoch
        while sample_count < train_epoch_size:  # loop over minibatches in the epoch
            # Get next mb
            data = reader_train.next_minibatch(min(mb_size,
                                                   train_epoch_size-sample_count),
                                               input_map=input_map)

            # Apply additional augmentation, optional
            if extra_aug:
                # Rotate image by 90 degrees
                # Flip image vert and/or horiz:  0 = ud; >0 = lr; <0 = ud + lr
                # Randomly swap channels: 'rgb'
                # Randomly drop channels: 'rgb'
                # Apply low-pass Gausian filter
                # Apply additive Gaussian noise
                data = data

            # Update model
            trainer.train_minibatch(data)

            # Update count
            sample_count += trainer.previous_minibatch_sample_count # n samples seen
            cumulative_count += sample_count

        # Summarize training at the end of each epoch
        trainer.summarize_training_progress()

#        # Store previous loss
#        running_loss.append(trainer.previous_minibatch_loss_average)

        # Evaluate test set accuracy
        if reader_test and epoch > 0:
            test_accuracy = evaluate_model.start(trainer, input_map,
                                                 label_var, reader_test,
                                                 test_epoch_size, mb_size)

            # Log average test set loss and prediction errror.
            if tensorboard_writer:
                tensorboard_writer.write_value("test_accuracy",
                                               test_accuracy,
                                               epoch)



        if data_struct['model_save_root']:
            trainer.save_checkpoint(os.path.join(data_struct['model_save_root'],
                                             data_struct['network_name'] + "_chkpt_{0}.dnn".format(epoch)))
        enable_profiler() # begin to collect profiler data after first epoch

    # Stop profiler, if enabled
    if profiler_dir:
        stop_profiler()

    return net, history

##
def clone():
    '''
    models.clone

    Returns a cloned network.
    '''

    net = 1

    return net

##
def load():
    '''
    models.load

    Returns a trained network
    '''


##
def test(net, # model_dict
         mapfile,
         df=None,
         output_filepath=None,
         reader=None,
         max_images=-1,
         model_suffix='',
         update_freq=50):
    '''
    models.test

    Returns a Padas df and test accuracy
    '''
    # Load mapfile into df
    if df:
        assert isinstance(df, pd.DataFrame), \
        'A Pandas Dataframe is required!'
    elif mapfile:
        assert os.path.isfile(mapfile), \
            'Mapfile {0:s} does not exist!'.format(mapfile)
        df, _, _ = mapfile_utils.read(mapfile)
    else:
        raise ValueError('A valid mapfile or dataframe are required!')


    # Set vars from dataframe
    num_classes = len(df['labels'].unique())
    epoch_size = df.shape[0]

    # Create mb_sources, if none given
    if reader is None:
        reader = default_reader.image(mapfile,
                                      transform_params,
                                      num_classes,
                                      num_channels=num_channels,
                                      height=image_height,
                                      width=image_width,
                                      mean_filepath=image_mean_filepath,
                                      is_training=is_training,
                                      use_mean_image=use_mean_image)

    # Set the output directory and filename, if given
    if model_suffix:
        output_filename =  'predictions_' + model_suffix + '.csv'
    else:
        output_filename =  'predictions.csv'

    if data_struct['model_save_root']:
        output_filepath = os.path.join(data_struct['model_save_root'], output_filename)
    elif data_struct['output_dir']:
        output_filepath = os.path.join(data_struct['output_dir'], output_filename)
    else:
        output_filepath = os.path.join(os.getcwd, output_filename)

    # Set input variables denoting the features and label data
    # input_var format is C,H,W
    input_var = C.input( (num_channels, image_height, image_width) )
    label_var = C.input( (num_classes) )

    # define mapping from reader streams to network inputs
    input_map = { input_var: reader_valid.streams.features }

    #TODO update for dataframe
    filepaths = 1
    labels =  1

    # process minibatches and evaluate the model
    prob_output = []
    pred_output = []
    label_output = []
    sample_count = 0
    correct_count = 0

    # Get the number of images in the mapfile
    if max_images > 0:
        num_images = min(num_images, max_images)

    # Evaluate over all test minibatches
    while sample_count < num_images:
        # get next image
        mb_data = reader_valid.next_minibatch(1, input_map = input_map)
        keys = list(mb_data.keys())
        if mb_data[keys[0]].shape==(1,1,3,224,224):
            img = mb_data[keys[0]]
            label = labels[sample_count] #mb_data[keys[1]]
        else:
            img = mb_data[keys[1]]
            label = labels[sample_count] #mb_data[keys[0]]

        # compute model output
        arguments = {net.arguments[0]:img}
        output = np.squeeze(net.eval(arguments))

        # return softmax probabilities
        prob = softmax(output).eval()
        pred = np.argmax(prob)

        # append to output list
        pred_output.append(pred)
        prob_output.append(prob[pred])
        label_output.append(label)

        # compare prediction with ground truth label
        if pred == label:
            correct_count += 1

        # add to counter
        sample_count += 1

        # Print updates
        if sample_count % update_freq == 0:
            print("Processed {0} samples ({1:.2%} correct)".format(sample_count,
                                                                   (float(correct_count) / sample_count)))



    # compute final accuracy
    accuracy = (float(correct_count) / sample_count)

    # print output if label given
    if label and sample_count==1:
        print('Predicted label:\t{:d}\nProbability:\t\t{:0.3f}\nTrue label:\t\t{:d}'.format(np.argmax(prob),
                                                                          prob[np.argmax(prob)],
                                                                          label))
    else:
         print("Processed {0} samples ({1:.2%} correct)".format(sample_count,accuracy))


    # create dataframe with compiled_results
    compiled_results = {
        'Y_hat':pred_output,
        'Prob': prob_output,
        'GT':label_output,
    }
    df = pd.DataFrame(compiled_results,
                      columns=['Y_hat','Prob', 'GT'])
    # Save to csv
    df.to_csv(output_filepath)

    return df, accuracy
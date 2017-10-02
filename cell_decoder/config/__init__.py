#  cell_decoder/config/__init__.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ======================================================================
'''

.. module:cell_decoder.config
    :synopsis: Configuration utilities and parameter classes.

'''

# Imports
import os
import copy
import numpy as np
import time

# Additional imports
import cntk as C
from cntk.learners import adadelta, adagrad, momentum_sgd, momentum_schedule, learning_rate_schedule, momentum_as_time_constant_schedule, rmsprop

## DataStructParameters class
class DataStructParameters():
    '''
    A DataStructParameters class for storing data and model information.
    '''
    def __init__(self,
                 debug_mode=True,
                 image_height=224,
                 image_mean_filepath=None,
                 image_width=224,
                 mapfile=None,
                 microns_per_pixel=0.25,
                 model_dict=None,
                 model_save_root=None,
                 num_channels=3,
                 pixel_mean=25,
                 profiler_dir=None,
                 random_seed=123456789,
                 scaling_factor=0.00390625,
                 tb_freq=10,
                 tb_log_dir='C:/TensorBoard_logs/cntk',
                 test_map=None,
                 text_labels=None,
                 train_map=None,
                 use_mean_image=False,
                 valid_map=None):
        # Either input one mapfile to convert into train/ test/
        # valid or each separately
        if mapfile:
            self.mapfile = mapfile
        else:
            self.train_map = train_map
            self.test_map = test_map
            self.valid_map = valid_map

        # Set the filepath for the text labels
        self.text_labels = text_labels

        # Set transform  crop parameters
        self.num_channels = num_channels
        self.image_height = image_height
        self.image_width = image_width

        # Set mean and scaling factors
        self.image_mean_filepath = image_mean_filepath
        self.pixel_mean = pixel_mean
        self.scaling_factor = scaling_factor
        self.use_mean_image = use_mean_image

        # Set log dirs
        self.tb_freq = tb_freq
        self.tb_log_dir = tb_log_dir
        self.profiler_dir = profiler_dir
        self.debug_mode = debug_mode
        self.microns_per_pixel = microns_per_pixel

        # Train/ save params
        self.model_dict = model_dict # Stores input_var, label_var, and net
        self.model_save_root = model_save_root

        # Set random state
        # TODO get seed automatically from random.org
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)

## Learning parameter class
class LearningParameters():
    OPTIMIZERS = ['sgd']
    '''
    A LearningParameters class for storing cntk learning parameters.
    '''
    def __init__(self,
                 epsilon=1e-3, # Adadelta
                 l2_reg_weight=1e-4, # CNTK L2 reg is per sample, like caffe
                 lr_per_mb=[ [0.01]*5 +[ 0.1]*30 + [0.01]*30 + [0.001]*25],
                 max_epochs=100,
                 mb_size=64,
                 momentum=0.9,
                 momentum_time_const=1,
                 optimizer='sgd',
                 rho=0.9, # Adadelta
                 loss='cross-entropy'):
        self.epsilon = epsilon
        self.l2_reg_weight = l2_reg_weight
        self.lr_per_mb = lr_per_mb
        self.loss_fn = loss
        self.max_epochs = max_epochs
        self.mb_size = mb_size
        self.momentum = momentum
        self.momentum_time_const = momentum_time_const
        self.optimizer = optimizer
        self.rho = rho
    
    #
    def compile(self,
                model_dict,
                epoch_size):
        '''
        compile

        Returns a dictionary of learning parameters
        '''
        # Error check
        valid_model_dict = ['input_var', 'label_var', 'net']
        if model_dict not in valid_model_dict:
            raise TypeError('Invalid model dictionary')
        
        if self.optimizer =='sgd':
            momentum_time_const = -self.mb_size/np.log(0.9)
            lr_per_sample  = [ lr/self.mb_size for lr in self.lr_per_mb]
            lr_schedule = learning_rate_schedule(lr_per_sample,
                                                 epoch_size=epoch_size,
                                                 unit=C.UnitType.sample)
            mm_schedule = momentum_as_time_constant_schedule(momentum_time_const)
            learner = momentum_sgd(model_dict['net'].parameters,
                                   lr_schedule,
                                   mm_schedule,
                                   l2_regularization_weight=learn_params['l2_reg_weight'])
            # Assign output dictionary
            learn_dict = {
                'learner': learner,
                'lr_per_mb': lr_per_mb,
                'lr_schedule': lr_schedule,
                'mb_size':self.mb_size,
                'mm_schedule': mm_schedule ,
                'momentum_time_const':momentum_time_const,
                'max_epochs':self.max_epochs
            }
            
        else:
            #TODO - complete this section
            raise RuntimeError('Section not complete')

        return learn_dict



## Transform parameter class
class TransformParameters():
    '''
    A TransformParameters class for storing cntk transform parameters.
    '''
    def __init__(self,
                 area_ratio=0.0,
                 aspect_ratio=1,
                 brightness_radius=0.3,
                 contrast_radius=0.3,
                 crop_size=0,
                 crop_type='randomside',
                 interpolations='linear',
                 jitter_type='uniratio',
                 saturation_radius=0.3,
                 side_ratio=(0.4, 0.875),
                 is_training=True):
        # Set attributes common to training/ testing
        self.interpolations = interpolations

        # Set train/ test specific attributes
        if is_training:
            self.crop_type = crop_type # "center", "randomside", "randomarea"
            self.crop_size = crop_size # crop_size (`int`, default 0):
            self.side_ratio = side_ratio # side_ratio (`float`, default 0.0)
            self.area_ratio = area_ratio # area_ratio (`float`, default 0.0):
            self.aspect_ratio = aspect_ratio # aspect_ratio (`float`, default 1.0):
            self.jitter_type = jitter_type # jitter_type (str, default 'none'):
            self.brightness_radius = brightness_radius
            self.contrast_radius = contrast_radius
            self.saturation_radius = saturation_radius

        else:
            # No jitter during the testing phase
            self.crop_type = 'multiview10' # 'center', 'multiview10'
            self.crop_size = 0
            self.side_ratio = np.median(side_ratio)
            self.area_ratio = 0.0
            self.aspect_ratio = 1
            self.jitter_type = None
            self.brightness_radius = 0.0
            self.contrast_radius = 0.0
            self.saturation_radius = 0.0


##
class ResNetParameters():
    '''
    A ResNetParameters class for storing default model parameters.
    '''
    # Class attributes
    RESNET_LAYERS = [18, 34, 50, 101, 152]
    NUM_STACK_LAYERS = {18:[2,1,1,2],
                        34:[3,3,5,2],
                        50:[2,3,5,2],
                        101:[2,3,22,2],
                        152:[2,7,35,2]}

    #
    def __init__(self,
                 allowed_layers=RESNET_LAYERS,
                 bias=True,
                 bn_time_const=4096,
                 dropout=True,
                 init=C.initializer.he_normal(),
                 model_name='ResNet50_ImageNet.model',
                 model_path='I:/Models/cntk/resnet',
                 num_filters=np.divide([64,128,256,512,1024,2048],2),
                 pad=True,
                 prefix='res',
                 resnet_layers=50,
                 num_stack_layers=NUM_STACK_LAYERS):

        # Error check
        if resnet_layers not in allowed_layers:
            raise ValueError('{0:d} is not a valid number of ResNet layers'.format(resnet_layers))

        assert len(num_filters)==6, \
            'num_filters must be a 6 element numpy vector!'

        # Set date_time
        date_time = time.strftime("%Y%m%d-%H%M")

        # Look up resnet num_stack_layers from class dict
        num_stack_layers = num_stack_layers[resnet_layers]
            
        # Update instance attributes
        self.bias = bias
        self.bn_time_const = bn_time_const
        self.dropout = dropout
        self.init = init
        self.model_name = '{0:s}_Res{1:d}.model'.format(date_time, resnet_layers) ## TODO edit
        self.model_path = model_path
        self.network_name = '{0:s}_Res{1:d}'.format(date_time, resnet_layers)
        self.num_filters = num_filters
        self.num_stack_layers = num_stack_layers
        self.pad = pad
        self.prefix = prefix
        self.resnet_layers = resnet_layers
        
        

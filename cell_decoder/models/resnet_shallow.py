#  cell_decoder/models/resnet_shallow.py
#
# TODO
#
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ======================================================================
'''
Creates deep residual networks with 18 or 34 layers.

Some functions within this module were adapted from Microsoft CNTK
tutorial and example files. The resnet blocks in this file
are Copyrighted (c) by Microsoft and licensed under the
MIT license.

----- Adapted from Microsoft CNTK Examples -----
----- MIT license -----
See the Microsoft MIT license at the link below:
      https://github.com/Microsoft/CNTK/blob/master/LICENSE.md
'''

# Imports
from __future__ import print_function

import os
import sys

# Array operations
import numpy as np

from cell_decoder import models

## Create 18 or 34 layer ResNet
def bottleneck(input_var,
               num_stack_layers,
               num_filters,
               data_struct,
               bias,
               bn_time_const,
               dropout,
               init,
               pad):
    '''create_resnet_bottleneck_mode

    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl

    ----- Adapted from Microsoft CNTK Examples -----
    ----- MIT license -----
    '''
    # Set variables
    num_classes = data_struct['num_classes']
    strides1x1 = (2,2)
    strides3x3 = (1,1)

    # Conv1
    conv_1 = conv_bn_relu(input_var, num_filters[0], (7,7), strides=(2,2), # input, out_size, kernel, stride,
                          init=init, bn_time_const=bn_time_const, name='conv1') # init, bn_time_const

    # Max pooling
    pool_1 = MaxPooling(filter_shape=(3,3), strides=2, pad=True, name='pool1')(conv_1)  # name='pool1'

    # conv2_x -- 2x [3x3, 64]
    r2_2 = resnet_basic_stack(pool_1, num_stack_layers[0], num_filters[0],
                              bn_time_const=bn_time_const,
                              bias=bias, pad=pad,
                              name='2')

    # conv3_x --  2x [3x3, 128]
    r3_1 = resnet_basic_inc(r2_1, num_filters[1], strides=(2,2), # strides
                            bn_time_const=bn_time_const, # bn_time constant
                            bias=bias, pad=pad,
                            name='res3a')
    r3_2 = resnet_basic_stack(r3_1, num_stack_layers[1], num_filters[1],
                              bn_time_const=bn_time_const,
                              bias=bias, pad=pad,
                              name='3')

    # conv4_x -- 2x [3x3, 256]
    r4_1 = resnet_basic_inc(r3_2, num_filters[2], strides=(2,2),
                                 bn_time_const=bn_time_const,
                                 bias=bias, pad=pad,
                                 name='res4a')
    r4_2 = resnet_basic_stack(r4_1, num_stack_layers[2], num_filters[2],
                              bn_time_const=bn_time_const,
                              bias=bias, pad=pad,
                              name='4')

    # conv5_x
    r5_1 = resnet_basic_inc(r4_2, num_filters[3], strides=(2,2),
                                 bn_time_const=bn_time_const,
                                 bias=bias, pad=pad,
                                 name='res5a')
    r5_2 = resnet_basic_stack(r5_1, num_stack_layers[3], num_filters[3],
                                   bn_time_const=bn_time_const,
                                   bias=bias, pad=pad,
                                   name='5')

    # GlobalAverage pooling -- rem from AvPool filter_shape=(7,7), strides=1,
    pool_5 = GlobalAveragePooling(name="pool_5")(r5_2)
    if dropout:
        drop_5 = Dropout(0.5, name='drop_5')(pool_5)
        net = Dense(num_classes, init=uniform(0.05), name="fc_final", activation=None)(drop_5)
    else:
        # Fully connected layer
        net = Dense(num_classes, init=uniform(0.05), name="fc_final", activation=None)(pool_5)

    return net

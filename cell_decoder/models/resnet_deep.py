#  cell_decoder/models/resnet_deep.py
#
# TODO
#
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ======================================================================
'''

.. module::cell_decoder.models.resnet_deep
    :synopsis: Creates deep residual networks with 50, 101, or 152 layers

Some functions within this module were adapted from Microsoft CNTK
tutorial and example files. The resnet blocks in this file
are Copyrighted (c) by Microsoft and licensed under the
MIT license.

See the Microsoft MIT license at the link below:
      https://github.com/Microsoft/CNTK/blob/master/LICENSE.md
'''

# Standard library imports
import os
import sys

# Array operations
import numpy as np

# Main CNTK imports
import cntk as C
from cntk.layers import AveragePooling, GlobalAveragePooling, MaxPooling, Dropout, Dense
from cntk.initializer import he_normal, he_uniform, glorot_normal,  glorot_uniform, uniform

# cell_decoder
from cell_decoder.models import resnet_blocks as res
# import conv_bv_relu, bottleneck_inc, bottleneck_stack, resnet_basic_inc, resnet_basic_stack

## Create 50, 101, or 152 layer ResNet
def bottleneck(input_var,
               num_stack_layers,
               num_filters,
               num_classes,
               bias,
               bn_time_const,
               dropout,
               init,
               pad):
    '''
    resnet_deep.bottleneck_model
    '''
    # Set variables
    num_classes = num_classes
    strides1x1 = (2, 2)
    strides3x3 = (1, 1)

    # Conv1
    conv_1 = res.conv_bn_relu(input_var, num_filters[0], (7, 7), strides=(2, 2), # input, out_size, kernel, stride,
                              init=init, bn_time_const=bn_time_const, name='conv1') # init, bn_time_const

    # Max pooling
    pool_1 = MaxPooling(filter_shape=(3, 3), strides=2, pad=True, name='pool1')(conv_1)  # name='pool1'

    # conv2_x
    r2_1 = res.bottleneck_inc(pool_1, num_filters[2], num_filters[0], # input, out_size, inter_out_size,
                              strides1x1=(1, 1), strides3x3=(1, 1), # strides
                              bn_time_const=bn_time_const, # bn_time constant
                              bias=bias, pad=pad,
                              name='res2a')
    r2_2 = res.bottleneck_stack(r2_1, num_stack_layers[0], num_filters[2], num_filters[0],
                                bn_time_const=bn_time_const,
                                bias=bias, pad=pad,
                                name='2')

    # conv3_x
    r3_1 = res.bottleneck_inc(r2_2, num_filters[3], num_filters[1],
                              strides1x1=strides1x1, strides3x3=strides3x3, # strides
                              bn_time_const=bn_time_const, # bn_time constant
                              bias=bias, pad=pad,
                              name='res3a')
    r3_2 = res.bottleneck_stack(r3_1, num_stack_layers[1], num_filters[3], num_filters[1],
                                bn_time_const=bn_time_const,
                                bias=bias, pad=pad,
                                name='3')

    # conv4_x
    r4_1 = res.bottleneck_inc(r3_2, num_filters[4], num_filters[2],
                              strides1x1=strides1x1, strides3x3=strides3x3,
                              bn_time_const=bn_time_const,
                              bias=bias, pad=pad,
                              name='res4a')
    r4_2 = res.bottleneck_stack(r4_1, num_stack_layers[2], num_filters[4], num_filters[2],
                                bn_time_const=bn_time_const,
                                bias=bias, pad=pad,
                                name='4')

    # conv5_x
    r5_1 = res.bottleneck_inc(r4_2, num_filters[5], num_filters[3],
                              strides1x1=strides1x1, strides3x3=strides3x3,
                              bn_time_const=bn_time_const,
                              bias=bias, pad=pad,
                              name='res5a')
    r5_2 = res.bottleneck_stack(r5_1, num_stack_layers[3], num_filters[5], num_filters[3],
                                bn_time_const=bn_time_const,
                                bias=bias, pad=pad,
                                name='5')

    # GlobalAverage pooling -- rem from AvPool filter_shape=(7,7), strides=1,
    pool_5 = GlobalAveragePooling(name="pool_5")(r5_2)
    if dropout:
        drop_5 = Dropout(0.5, name='drop_5')(pool_5)
        net = Dense(num_classes, init=uniform(np.divide(1, num_classes)),
                    name="fc_final", activation=None)(drop_5)
    else:
        # Fully connected layer
        net = Dense(num_classes, init=uniform(np.divide(1, num_classes)),
                    name="fc_final", activation=None)(pool_5)

    return net


## Create 50, 101, or 152 layer ResNet
def basic(input_var,
          num_stack_layers,
          num_filters,
          num_classes,
          bias,
          bn_time_const,
          dropout,
          init,
          pad):
    '''
    resnet_deep.basic_model
    '''
    # Set variables
    num_classes = num_classes
    strides1x1 = (2, 2)
    strides3x3 = (1, 1)

    # Conv1
    conv_1 = res.conv_bn_relu(input_var, num_filters[0], (7, 7), strides=(2, 2), # input, out_size, kernel, stride,
                              init=init, bn_time_const=bn_time_const, name='conv1') # init, bn_time_const

    # Max pooling
    pool_1 = MaxPooling(filter_shape=(3,3), strides=2, pad=True, name='pool1')(conv_1)  # name='pool1'

    # conv2_x
    r2_1 = res.bottleneck_inc(pool_1, num_filters[2], num_filters[0], # input, out_size, inter_out_size,
                              strides1x1=(1, 1), strides3x3=(1, 1), # strides
                              bn_time_const=bn_time_const, # bn_time constant
                              bias=bias, pad=pad,
                              name='res2a')
    r2_2 = res.bottleneck_stack(r2_1, num_stack_layers[0], num_filters[2], num_filters[0],
                                bn_time_const=bn_time_const,
                                bias=bias, pad=pad,
                                name='2')

    # conv3_x
    r3_1 = res.bottleneck_inc(r2_2, num_filters[3], num_filters[1],
                              strides1x1=strides1x1, strides3x3=strides3x3, # strides
                              bn_time_const=bn_time_const, # bn_time constant
                              bias=bias, pad=pad,
                              name='res3a')
    r3_2 = res.bottleneck_stack(r3_1, num_stack_layers[1], num_filters[3], num_filters[1],
                                bn_time_const=bn_time_const,
                                bias=bias, pad=pad,
                                name='3')

    # conv4_x
    r4_1 = res.bottleneck_inc(r3_2, num_filters[4], num_filters[2],
                              strides1x1=strides1x1, strides3x3=strides3x3,
                              bn_time_const=bn_time_const,
                              bias=bias, pad=pad,
                              name='res4a')
    r4_2 = res.bottleneck_stack(r4_1, num_stack_layers[2], num_filters[4], num_filters[2],
                                bn_time_const=bn_time_const,
                                bias=bias, pad=pad,
                                name='4')

    # conv5_x
    r5_1 = res.bottleneck_inc(r4_2, num_filters[5], num_filters[3],
                              strides1x1=strides1x1, strides3x3=strides3x3,
                              bn_time_const=bn_time_const,
                              bias=bias, pad=pad,
                              name='res5a')
    r5_2 = res.bottleneck_stack(r5_1, num_stack_layers[3], num_filters[5], num_filters[3],
                                bn_time_const=bn_time_const,
                                bias=bias, pad=pad,
                                name='5')

    # GlobalAverage pooling -- rem from AvPool filter_shape=(7,7), strides=1,
    pool_5 = GlobalAveragePooling(name="pool_5")(r5_2)
    if dropout:
        drop_5 = Dropout(0.5, name='drop_5')(pool_5)
        net = Dense(num_classes, init=uniform(np.divide(1, num_classes)), name="fc_final", activation=None)(drop_5)
    else:
        # Fully connected layer
        net = Dense(num_classes, init=uniform(np.divide(1, num_classes)), name="fc_final", activation=None)(pool_5)

    return net

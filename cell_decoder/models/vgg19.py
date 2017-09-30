#  cell_decoder/models/vgg16.py
#
# TODO --packaged with cell_decoder for convenience
#
# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ======================================================================
'''
Creates a convolutional neural network with the VGG 19 architecture.

REF

Some functions within this module were adapted from Microsoft CNTK
tutorial and example files. The resnet blocks in this file
are Copyrighted (c) by Microsoft and licensed under the
MIT license.

See the Microsoft MIT license at the link below:
      https://github.com/Microsoft/CNTK/blob/master/LICENSE.md
'''

# Imports
from __future__ import print_function

import os
import sys

# Array operations
import numpy as np

from cntk_utils import utils

##
def create_vgg19():
    '''
    cntk model for vgg19
    
    
    Network architecture as described in:
         https://arxiv.org/pdf/1409.1556v6.pdf
    
    ----- Adapted from Microsoft CNTK Examples -----
    ----- MIT license -----
    '''
#     with default_options(activation=None, pad=True, bias=True):
#        z = Sequential([
#            # we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU)
#            For(range(2), lambda i: [
#                Convolution2D((3,3), 64, name='conv1_{}'.format(i)),
#                Activation(activation=relu, name='relu1_{}'.format(i)),
#            ]),
#            MaxPooling((2,2), (2,2), name='pool1'),
#
#            For(range(2), lambda i: [
#                Convolution2D((3,3), 128, name='conv2_{}'.format(i)),
#                Activation(activation=relu, name='relu2_{}'.format(i)),
#            ]),
#            MaxPooling((2,2), (2,2), name='pool2'),
#
#            For(range(4), lambda i: [
#                Convolution2D((3,3), 256, name='conv3_{}'.format(i)),
#                Activation(activation=relu, name='relu3_{}'.format(i)),
#            ]),
#            MaxPooling((2,2), (2,2), name='pool3'),
#
#            For(range(4), lambda i: [
#                Convolution2D((3,3), 512, name='conv4_{}'.format(i)),
#                Activation(activation=relu, name='relu4_{}'.format(i)),
#            ]),
#            MaxPooling((2,2), (2,2), name='pool4'),
#
#            For(range(4), lambda i: [
#                Convolution2D((3,3), 512, name='conv5_{}'.format(i)),
#                Activation(activation=relu, name='relu5_{}'.format(i)),
#            ]),
#            MaxPooling((2,2), (2,2), name='pool5'),
#
#            Dense(4096, name='fc6'),
#            Activation(activation=relu, name='relu6'),
#            Dropout(0.5, name='drop6'),
#            Dense(4096, name='fc7'),
#            Activation(activation=relu, name='relu7'),
#            Dropout(0.5, name='drop7'),
#            Dense(num_classes, name='fc8')
#            ])(input)

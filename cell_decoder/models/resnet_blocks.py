# cell_decoder/models/create_model.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under a BSD "two-clause" license.
# ======================================================================
'''
Cell DECODER neural network building blocks

Some functions within this module were adapted from Microsoft CNTK
tutorial and example files. The resnet blocks in this file
are Copyrighted (c) by Microsoft and licensed under the
MIT license.

See the Microsoft MIT license at the link below:
      https://github.com/Microsoft/CNTK/blob/master/LICENSE.md
'''

# ====================== Modules to build ResNets ======================
# These functions include code that was adapted from the CNTK tutorials
# for cifar10 in python and brainscript. The functions have been modified.
# ======================================================================

# Imports
import os
import sys
import numpy as np

# Main CNTK imports
import cntk as C
from cntk import combine, load_model, softmax, CloneMethod, Trainer, UnitType
from cntk.initializer import he_normal, he_uniform, glorot_normal,  glorot_uniform, uniform
from cntk.layers import AveragePooling, GlobalAveragePooling, BatchNormalization, Convolution, Dense, Dropout, MaxPooling, MaxUnpooling
from cntk.ops import element_times, relu
from cntk.logging.graph import find_by_name, get_node_outputs


## Convolution with batch normalization
def conv_bn(input_var, num_filters, filter_size, strides=(1,1), # input, output_size, kernel, stride
            init=he_normal(), spatial_rank=1, bn_time_const=4096,  # init, map_rank, bn_time_const
             prefix='', bias=True, pad=True, name=None): # name
    '''conv_bv

    Accepts X and returns y.
    '''
    # Convolutional layer
    c = Convolution(filter_size, num_filters, activation=None,
                    init=init, pad=pad, strides=strides, bias=bias,
                    name=prefix + name)(input_var)

    # Batch normalization
    r = BatchNormalization(map_rank=spatial_rank,
                           normalization_time_constant=bn_time_const,
                           use_cntk_engine=False,
                           name='bn'+name)(c)

    return r


## Convolutional with batch normalization and ReLU
def conv_bn_relu(input_var, num_filters, filter_size, strides=(1,1), # input, out_size, kernel, stride
                 init=he_normal(), spatial_rank=1, bn_time_const=4096, # init, map_rank, bn_time_const
                 prefix='', bias=True, pad=True, name=''):
    '''conv_bv_relu

    Accepts X and returns Y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    # Call the conv_bn function
    r = conv_bn(input_var, num_filters, filter_size, strides=strides,
                init=init, spatial_rank=spatial_rank,
                bn_time_const=bn_time_const,
                bias=bias, pad=pad,
                name=name)
    # Relu of the result from conv_bn
    r = relu(r, name=prefix+name+'_relu')

    return r


## Basic resnet block
# The basic Resnet block consistes of two 3x3 convolutions,
# which is added to the original input of the block
def basic(input_var, num_filters, bn_time_const=4096,
                  prefix='',bias=True, pad=True, name=''):
    '''basic

    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    # Conv
    c1 = conv_bn_relu(input_var, num_filters, (3,3), strides=(1,1),
                      bn_time_const=bn_time_const,
                      prefix=prefix, bias=bias, pad=pad,
                      name=name)
    c2 = conv_bn(c1, num_filters, (3,3), strides=(1,1),
                 bn_time_const=bn_time_const,
                 prefix='', bias=bias, pad=pad,
                 name=name)

    # Add to original
    p = C.ops.plus(c2, input_var, name=name)
    return relu(p, name=name+'_relu')


## Feature map reduction
# This block contains two 3x3 convolutions with stride,
# which is added to the original input with 1x1 convolution and stride
def basic_inc(input_var, num_filters, strides=(2,2),
                     bn_time_const=4096, prefix='', bias=True,
                     pad=True, name=''):
    '''basic_inc

    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    # Default filter size for ResNet
    filter_size=(3,3)

    # Conv
    c1 = conv_bn_relu(input_var, num_filters, filter_size, strides=strides,
                      bn_time_const=bn_time_const,
                      prefix=prefix, bias=bias, pad=pad,
                      name=name+'_branch2a')
    c2 = conv_bn(c1, num_filters, filter_size, strides=(1,1),
                 prefix=prefix,bias=bias, pad=pad,
                 name=name+'_branch2b')

    # Shortcut
    s  = conv_bn(input_var, num_filters, (1,1),  strides=strides,
                 bn_time_const=bn_time_const,
                 prefix=prefix, bias=bias, pad=pad,
                 name=name+'_branch2c')

    p  = C.ops.plus(c2, s, name=name)
    return relu(p,name=name+'_relu')


## Resnet bottleneck block
# This block reduces the computation by replacing the two 3x3 convs
# with a 1x1 conv, bottlenecked to the inter_out_channels (output features),
# and is followed by a 3x3 conv and a 1x1 conv
def bottleneck(input_var, num_filters, inter_out_channels,
                      bn_time_const=4096, prefix='', bias=True, pad=True,
                      name=''):
    '''resnet_bottleneck_inc

    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    # Conv
    # Resnet bottleneck adds '_branch2x'
    # 1x1 conv
    c1 = conv_bn_relu(input_var, inter_out_channels, (1,1), strides=(1,1),
                      bn_time_const=bn_time_const,
                      prefix=prefix, bias=bias, pad=pad,
                      name=name+'_branch2a')
    # 3x3 conv
    c2 = conv_bn_relu(c1, inter_out_channels, (3,3), strides=(1,1),
                      prefix=prefix, bn_time_const=bn_time_const,
                      bias=bias, pad=pad,
                      name=name+'_branch2b')
    # 1x1 conv
    c3 = conv_bn(c2, num_filters, (1,1), strides=(1,1),
                 bn_time_const=bn_time_const,
                 prefix=prefix, bias=bias, pad=pad,
                 name=name+'_branch2c')

    p  = C.ops.plus(c3, input_var, name=name)
    return relu(p, name=name+'_relu')


## Feature reduction with bottleneck
# One can reduce the size either at the first 1x1 convolution by specifying
# "strides1x1=(2:2)" (original paper), or at the 3x3 convolution by specifying
# "strides3x3=(2:2)" (Facebook re-implementation).
# See Figure 5 from He et al 2015
def bottleneck_inc(input_var, num_filters, inter_out_channels,
                          strides1x1=(2,2), strides3x3=(1,1),
                          bn_time_const=4096, prefix='', bias=True, pad=True,
                          name=''):
    '''bottleneck_inc

    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    # Conv
    # 1x1 conv
    c1 = conv_bn_relu(input_var, inter_out_channels, (1,1), strides=strides1x1,
                      bn_time_const=bn_time_const,
                      prefix=prefix, bias=bias, pad=pad,
                      name=name+'_branch2a')
    # 3x3 conv
    c2 = conv_bn_relu(c1, inter_out_channels, (3,3), strides=strides3x3,
                      bn_time_const=bn_time_const,
                      prefix=prefix, bias=bias, pad=pad,
                      name=name+'_branch2b')
    # 1x1 conv
    c3 = conv_bn(c2, num_filters, (1,1), strides=(1,1),
                 bn_time_const=bn_time_const,
                 prefix=prefix, bias=bias, pad=pad,
                 name=name+'_branch2c')

    # Shortcut
    strides_shortcut = np.multiply(strides1x1,strides3x3)
    s  = conv_bn(input_var, num_filters, (1,1),  strides=strides_shortcut,
                 bn_time_const=bn_time_const,
                 bias=bias, pad=pad,
                 name=name+'_branch1')

    p  = C.ops.plus(c3, s, name=name)
    return relu(p, name=name+'_relu')


## Create a basic ResNet stack
def basic_stack(input_var, num_stack_layers, num_filters,
                       prefix='', bn_time_const=4096, bias=True, pad=True,
                       name=None):
    '''basic_stack

    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    assert (num_stack_layers >= 0)
    l = input_var

    node_suffix = ['a','b','c','d','e','f','g','h','i','j']
    for n_layer in range(num_stack_layers):
        l = basic(l, num_filters,
                  name=name+'{0:s}'.format(node_suffix[n_layer]))

    return l


## Create a bottleneck ResNet stack
def bottleneck_stack(input_var, num_stack_layers, num_filters, inter_out_channels,
                            bn_time_const=4096, prefix='', bias=True, pad=True,
                            name=''):
    '''bottleneck_stack

    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    assert (num_stack_layers >= 0)
    l = input_var

    node_suffix = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','bb','cc','dd','ee','ff','gg','hh','ii','jj','kk','ll','mm','nn','oo','pp','qq','rr','ss','tt','uu','vv','ww','xx','yy','zz']
    for n_layer in range(num_stack_layers):
        l = bottleneck(l, num_filters, inter_out_channels,
                       bn_time_const=bn_time_const,
                       prefix=prefix, bias=bias, pad=pad,
                       name=name+'{0:s}'.format(node_suffix[n_layer]))

    return l




### Create a full resnet model
#def bottleneck_model(data_struct,
#                            resnet_layers=50,
#                            bn_time_const=4096,
#                            init=he_normal(),
#                            clone=False,
#                            freeze=True,
#                            node='conv1_relu',
#                            model_name='ResNet50_ImageNet.model',
#                            model_path='I:/Models/cntk/resnet',
#                            prefix='res',
#                            bias=True,
#                            pad=True,
#                            dropout=False):
#    '''
#    bottleneck_model
#
#    '''
#    # Set input
#    input_var = C.input_variable(
#        (data_struct['num_channels'],
#         data_struct['image_height'],
#         data_struct['image_width'])
#    )
#
#    # Scale 0-1
#    input_scaled = C.ops.element_times(input_var,
#                                       C.ops.constant(0.00390625),
#                                       name="input_scaled")
#    # Subtract scaled mean (30/255)
#    input_mean_sub = C.ops.minus(input_scaled,
#                                 C.ops.constant(0.1000), # RGB [25.908, 21.671, 19.730]
#                                 name="input_mean_sub")
#
#    # Set num_classes
#    label_var = C.input_variable(
#        data_struct['num_classes']
#    )
#
#    # Set filter cmap and num_stack_layers
#    num_filters = np.divide([64,128,256,512,1024,2048],2)
#
#    # Create resnet stack
#    if resnet_layers == 18:
##        raise RuntimeError('Not complete')
#        num_stack_layers = [2,1,1,2]
#        net = create_imagenet_model_small(input_mean_sub, num_stack_layers, num_filters,data_struct,
#                                bias, bn_time_const, dropout,  init,  pad)
#
#    elif resnet_layers == 34:
##        raise RuntimeError('Not complete')
#        num_stack_layers = [3,3,5,2]
#        net = create_imagenet_model_small(input_mean_sub, num_stack_layers,
#                                          num_filters,data_struct,
#                                          bias, bn_time_const, dropout, init,
#                                          pad)
#
#    elif resnet_layers == 50:
#        num_stack_layers = [2,3,5,2]
#        net = create_imagenet_model(input_mean_sub, num_stack_layers,
#                                    num_filters,data_struct, bias,
#                                    bn_time_const, dropout, init,
#                                    pad)
#    elif resnet_layers == 101:
#        num_stack_layers = [2,3,22,2]
#        net = create_imagenet_model(input_mean_sub, num_stack_layers,
#                                    num_filters,data_struct, bias,
#                                    bn_time_const, dropout, init,
#                                    pad)
#
#    elif resnet_layers == 152:
#        num_stack_layers = [2,7,35,2]
#        net = create_imagenet_model(input_mean_sub, num_stack_layers,
#                                    num_filters,data_struct, bias,
#                                    bn_time_const, dropout, init,
#                                    pad)
#    else:
#        raise RuntimeError('Unknown number of resnet layers')
#
#    # Update data_struct
#    data_struct['input_var'] = input_var
#    data_struct['label_var'] = label_var
#    data_struct['network_name'] = 'Res{0:d}'.format(resnet_layers)
#    data_struct['model_name'] = 'Res{0:d}_ImageNet.model'.format(resnet_layers)
#    data_struct['model'] = net
#
#    return data_struct
#
#
### Clone a ResNet model and append a new FC layer for transfer learning
#def clone_resnet_model(data_struct, pred_name='pool_5',
#                       model_path=None, model_name='none',
#                       freeze=False):
#    # Set variables
#    input_var = C.input_variable(
#        (data_struct['num_channels'],
#        data_struct['image_height'],
#         data_struct['image_width'])
#    )
#    label_var = C.input_variable(
#        data_struct['num_classes']
#    )
#
#    num_classes = data_struct['num_classes']
#
#    model_path = os.path.normpath(os.path.join(data_struct['model_path'],
#                                               data_struct['model_name']))
#    print('Loading model {}'.format(model_path))
#    base_model = load_model(model_path)
#    feature_node = base_model.arguments[0] #find_by_name(base_model, 'data')
#    last_node = find_by_name(base_model, pred_name)
#
#    # Clone the desired layers with or without fixed weights
#    if freeze:
#        print('Cloning layers with fixed weights')
#        cloned_layers = combine([last_node.owner]).clone(CloneMethod.freeze,
#                                                         {feature_node: C.placeholder(name='features')})
#    else:
#        print('Cloning layers with learnable weights')
#        cloned_layers = combine([last_node.owner]).clone(CloneMethod.clone,
#                                                         {feature_node: C.placeholder(name='features')})
#
#    # Set input for cloned output
#    cloned_out = cloned_layers(input_var)
#
#    # New fully connected layer
#    net = Dense(num_classes, activation=None, name='prob')(cloned_out)
#
#     # Update data_struct
#    data_struct['input_var'] = input_var
#    data_struct['label_var'] = label_var
#    data_struct['model'] = net
#    return data_struct

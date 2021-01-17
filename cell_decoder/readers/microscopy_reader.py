# microscopy_reader.py -- a module within cntk_utils
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under a BSD "two-clause" license.
# ==============================================================================
#
# The following reader was adapted from the Microsoft CNTK FERPlus reader.
# These files are Copyrighted (c) by Microsoft and licensed under the MIT license.
# See the Microsoft MIT license at the link below:
#      https://github.com/Microsoft/FERPlus/src/ferplus.py
#
# Also see:
# https://cntk.ai/pythondocs/extend.html#user-defined-minibatch-sources
# https://docs.microsoft.com/en-us/cognitive-toolkit/BrainScript-and-Python---Understanding-and-Extending-Readers

# Standard lib imports
from __future__ import print_function

# Imports
import sys
import os
import csv
import numpy as np
import logging
import random as rnd
from collections import namedtuple

# Additional imports
import cv2
from cntk.io import UserMinibatchSource, StreamInformation, MinibatchData
import img_utils # Update
#from cntk.logging import *
#from cntk.debugging import set_computation_network_trace_level


#from rect_util import Rect # Update
#import matplotlib.pyplot as plt # Do I need this?

## Summarize the data
#def display_summary(train_data_reader, val_data_reader, test_data_reader):
#    '''
#    Summarize the data in a tabular format.
#    '''
#    emotion_count = train_data_reader.emotion_count # TODO
#    emotion_header = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
#
#    logging.info("{0}\t{1}\t{2}\t{3}".format("".ljust(10), "Train", "Val", "Test"))
#    for index in range(emotion_count):
#        logging.info("{0}\t{1}\t{2}\t{3}".format(emotin_header[index].ljust(10), 
#                     train_data_reader.per_emotion_count[index], 
#                     val_data_reader.per_emotion_count[index], 
#                     test_data_reader.per_emotion_count[index]))


## Define custom reader parameters        
class MicroscopyReaderParameters():
    '''
    Microscopy reader parameters
    '''
    def __init__(self, target_size, width=224, height=224,
                 channels=3, training_mode = 'majority', 
                 determinisitc = False, is_training= True):
        self.target_size = target_size
        self.width = width
        self.height = height
        self.channels = channels
        self.training_mode = training_mode
        self.determinisitc = determinisitc
        self.is_training = is_training


## Define custom reader
class MicroscopyReader(object): # UserMinibatchSource
    '''
    A CNTK custom reader for fluorescence microscopy images that
    supports data augmentation and multiple traning modes.
    
    Training modes:
    'majority' # default
    'crossentropy'
    'multi_target'
    'probability'
    '''
    @classmethod
    def create(cls, base_folder, sub_folders, label_file_name, parameters):
        '''
        Factory function that create an instance of FERPlusReader and load the data form disk.
        '''
        reader = cls(base_folder, sub_folders,
                     label_file_name, parameters)
        reader.load_folders(parameters.training_mode)
        return reader
    
    ##
    def __init__(self, map_file, parameters):
        '''
        Load the map file containing filepaths and labels.
        '''
        self.map_file = map_file
        self.width = parameters.width
        self.height = parameters.height
        self.channels = parameters.channels
#        self.num_classes     = 
        self.is_training = parameters.is_training

        # data augmentation parameters.determinisitc
        if parameters.determinisitc:
            self.max_shift = 0.0
            self.max_scale = 1.0
            self.max_angle = 0.0
            self.max_skew  = 0.0
            self.do_flip   = False
            self.do_rotate = False
            self.swap_ch   = ''
            self.drop_ch   = ''            
        else:
            self.max_shift = 0.08
            self.max_scale = 1.05
            self.max_angle = 20.0
            self.max_skew  = 0.05
            self.do_flip   = True
            self.do_rotate = True
            self.swap_ch   = 'rg'
            self.drop_ch   = 'rg'
        
        self.data              = None
        self.batch_start       = 0
        self.indices           = 0
        
        # I moved text from here to scratch
        self.A, self.A_pinv = imgu.compute_norm_mat(self.width, self.height)

        with open(map_file) as csv_file:
            data = csv.reader(csv_file)
            for row in data:
                self.video_files.append(os.path.join(map_file_dir, row[0]))
                target = [0.0] * self.label_count
                target[int(row[1])] = 1.0
                self.targets.append(target)
            
        self.indices = np.arange(len(self.video_files))
        
        if self.is_training:
            np.random.shuffle(self.indices)
        self.epoch_size = min(len(self.video_files), limit_epoch_size)
    
        # Shuffle images
        if self.is_training:
            np.random.shuffle(self.indices)

    ##
    def stream_infos(self):
        """
        stream_infos():
        returns a list of StreamInformation instances that describe the
        streams the minibatch source is providing.
        See https://cntk.ai/pythondocs/extend.html#user-defined-minibatch-sources
        """
        return [self.fsi, self.lsi]
    
    
    ## Check whether there are more minibatchs
    def has_more(self):
        '''
        Return True if there are more mini-batches.
        '''
        return self.batch_start < len(self.data)
    
    ## Restart epoch (training mode only)
    def reset(self):
        '''
        Start from beginning for the new epoch.
        '''
        self.batch_start = 0

    ## Return the total number of images read
    def size(self):
        '''
        Return the number of images read by this reader.
        '''
        return len(self.data)

    ## Return the next minibatch
    def next_minibatch(self, batch_size):
        '''
        Return the next mini-batch. Data augmentation is during
        each mini-batch construction.
        '''
        data_size = len(self.data)
        batch_end = min(self.batch_start + batch_size, data_size)
        current_batch_size = batch_end - self.batch_start
        
        if current_batch_size < 0:
            raise Exception('The end of the training data has been reached.')
        
        inputs  = np.empty(shape=(current_batch_size, 1, self.width, self.height), dtype=np.float32)
        targets = np.empty(shape=(current_batch_size, self.emotion_count), dtype=np.float32)
        
        # Pre-allocate the random vars to set which transforms to apply
        
        # Flip switch (bernoulli, p = 0.5)
        # Flip type (ud vs lr vs ud + lr, multinoulli)
        if self.do_flip:
            do_flip = np.random.randint(2, size=(batch_end,1), dtype=bool)
            flip_type = np.random.randint(-1,2, size=(batch_end,1), dtype=int)
        else:
            do_flip  = np.zeros((batch_end,1),
                                dtype=bool)
            flip_type = np.random.randint(2,
                                          size=(batch_end,1),
                                          dtype=bool)
        
        # Rotation switch (bernoulli, p = 0.5)
        if self.do_rot:
            do_rot = np.random.randint(2, size=(batch_end,1), dtype=bool)
            rot_deg = np.random.randint(1,high=5, size=(batch_end,1), dtype=int)
        else:
            do_rot = np.zeros((batch_end,1), dtype=bool)
            rot_deg = np.zeros((batch_end,1), dtype=int)
        
        # Loop over all images in minibatch
        for idx in range(self.batch_start, batch_end):
            index = self.indices[idx]
            distorted_image = imgu.distort_img(self.data[index][1],
                                               self.data[index][3],
                                               self.width, 
                                               self.height, 
                                               self.max_shift, 
                                               self.max_scale, 
                                               self.max_angle, 
                                               self.max_skew, 
                                               do_flip[idx],
                                               flip_type[idx],
                                               do_rot[idx],
                                               rot_deg[idx])
            final_image = imgu.preproc_img(distorted_image,
                                           A=self.A,
                                           A_pinv=self.A_pinv)
            
            inputs[idx-self.batch_start]    = final_image
            targets[idx-self.batch_start,:] = self._process_target(self.data[index][2])
        
        self.batch_start += current_batch_size
        return inputs, targets, current_batch_size

# Also see
# https://github.com/Microsoft/CNTK/blob/master/Examples/Video/GettingStarted/Python/Conv3D_UCF11.py
    def _process_target(self, target):
        '''
        Based on https://arxiv.org/abs/1608.01041 the target depend on the training mode.
        Majority or crossentropy: return the probability distribution generated by "_process_data"
        Probability: pick one emotion based on the probability distribtuion.
        Multi-target: 
        '''
        if self.training_mode in ['majority', 'crossentropy']: 
            return target
        elif self.training_mode == 'probability': 
            idx             = np.random.choice(len(target), p=target) 
            new_target      = np.zeros_like(target)
            new_target[idx] = 1.0
            return new_target
        elif self.training_mode == 'multi_target': 
            new_target = np.array(target) 
            new_target[new_target>0] = 1.0
            epsilon = 0.001     # add small epsilon in order to avoid ill-conditioned computation
            return (1-epsilon)*new_target + epsilon*np.ones_like(target)

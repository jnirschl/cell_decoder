# cell_decoder/io/__init__.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ======================================================================
'''

.. module::cell_decoder.io
    :synopsis: Input-output utilities.

'''

# Imports
import os
import copy
import numpy as np
import pandas as pd

#
import cv2

# Additional imports
import cntk as C

import cell_decoder as c_decoder
from cell_decoder import models
from cell_decoder.config import DataStructParameters, LearningParameters, ResNetParameters, TransformParameters
from cell_decoder.img import compute_mean, img_utils
from cell_decoder.io import mapfile_utils
from cell_decoder.readers import default_reader, microscopy_reader
from cell_decoder.visualize import plot


## Create a data_struct class
class DataStruct:
    '''
    A DataStruct class for storing cell_decoder information to
    train and evaluate cntk models.

    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    # Class attributes
    RESNET_LAYERS = [18, 34, 50, 101, 152]
    READERS = ['default','microscopy']

    # Initialize class instance attributes
    def __init__(self,
                 mapfile,
                 parameters=None,
                 frac_sample=0.01):
        '''
        Read the mapfile and set data_struct parameters.

        '''
        assert os.path.isfile(mapfile), \
            'Mapfile {0:s} does not exist!'.format(mapfile)#
        assert (isinstance(parameters, DataStructParameters.__class__) or \
                parameters is None), \
                'Parameters must be a DataStructParameters class.'

        # Get parameters if none given
        if parameters is None:
            parameters = DataStructParameters(mapfile)

        # Copy parameters from DataStructParameters (parent class)
        for k, v in parameters.__dict__.items():
            self.__dict__[k] = copy.deepcopy(v)

        # Read mapfile
        df, mapfile_root, mapfile_name = mapfile_utils.read(mapfile,
                                                            frac_sample=frac_sample)

        # Update instance attributes
        self.num_classes = int(df['label'].max() + 1)
        self.mapfile_root = mapfile_root
        self.mapfile_name = mapfile_name
        self.epoch_size = df.shape[0]

    ##
    def read_mapfile(self):
        '''
        DataStruct.read_mapfile()
        
        Returns a the mapfile as a Pandas dataframe 
        '''

        df, _, _ = mapfile_utils.read(self.mapfile)

        return df
    
    ##
    def compute_mean(self,
                     savepath=None,
                     filename=None,
                     data_aug=True,
                     debug_mode=True,
                     save_img=True,
                     nargout=False):
        '''
        DataStruct.compute_mean()

        Computes the mean image and RGB pixel values
        for a given mapfile and saves a PNG and OpenCV
        XML file of the mean image.

        Optional: Returns the mean image as an RGB numpy array.
        '''
        # Set savepath
        if savepath is None:
            savepath = os.path.join(os.path.dirname(compute_mean.__file__),
                                    '../../meanfiles/')
            savepath = os.path.normpath(savepath)

        # Set filename
        if filename is None:
            filename = self.mapfile_name

        image_mean = compute_mean.image(self.mapfile,
                                        savepath=savepath,
                                        filename=filename,
                                        data_aug=data_aug,
                                        save_img=save_img)

        # Store path to mean image xml, png
        self.image_mean_filepath = os.path.join(savepath,
                                                filename.replace('.png','.xml'))

        # Store mean pixel values (BGR)
        self.pixel_mean = np.mean(np.mean(image_mean, axis=0), axis=0)

        # Optional output argument
        if nargout:
            # Convert to RGB
            image_mean = cv2.cvtColor(image_mean, cv2.COLOR_BGR2RGB)

            # Normalize image
            image_mean = img_utils.to_float(image_mean)
            
            return image_mean

    ##
    def cvpartition(self,
                    k_fold=5,
                    savepath='',
                    held_out_frac=0.1,
                    held_out_n=100,
                    random_state=1):
        '''
        DataStruct.cvpartition()

        Partitions a mapfile into a training and held-out test
        data set. The training data set is split into k-fold for
        cross-validation.

        '''
        # Setup cross validation
        df, df_held_out = mapfile_utils.crossval(self.mapfile,
                                                 k_fold=k_fold,
                                                 held_out_test=held_out_test,
                                                 held_out_n=held_out_n,
                                                 savepath=savepath,
                                                 random_seed=random_state)

        # Set

        return df

    ##
    def create_reader(self,
                      transform_params,
                      held_out_n=100,
                      held_out_test=True,
                      is_training=True,
                      random_seed=None,
                      reader='default',
                      savepath=os.getcwd(),
                      k_fold=5,
                      allowed_readers=READERS):
        '''
        DataStruct.create_reader(TransformParams)

        Creates CNTK Minibatch Source readers for training
        and testing mapfiles.
        '''
        if reader not in allowed_readers:
            raise ValueError('Invalid reader {0:s}!'.format(reader))

        # Create minibatch source from DataStruct instance
        if reader.lower()=='default':
            mb_source = default_reader.image(self.mapfile,
                                             transform_params,
                                             num_classes=self.num_classes,
                                             num_channels=self.num_channels,
                                             height=self.image_height,
                                             width=self.image_width,
                                             mean_filepath=self.image_mean_filepath,
                                             is_training=is_training,
                                             use_mean_image=self.use_mean_image)
        elif reader.lower()=='microscopy':
            raise RuntimeError('This section not complete!')

        return mb_source

    ##
    def create_model(self,
                     model_parameters=None,
                     allowed_layers=RESNET_LAYERS):
        '''
        DataStruct.create_model()

        Returns a dictionary "model_dict" with the keys:
            input_var, label_var, net, and num_classes.
        '''
        #TODO
        #        if self.resnet_layers not in allowed_layers:
        #            raise ValueError('Invalid number of resnet layers {0:d}!'.format(reader))

        # Validate model_parameters
        valid_class = ResNetParameters().__class__
        if (model_parameters and isinstance(model_parameters, valid_class)):
            valid_model_params = ResNetParameters()

            for elem in model_parameters.__dict__:
                if elem not in valid_model_params.__dict__.keys():
                    raise TypeError('Invalid model parameter')

        else:
            print('Using default model parameters.')
            model_parameters = ResNetParameters()

        # Call models.create subfunction
        model_dict = models.create(model_parameters,
                                   num_classes=self.num_classes,
                                   pixel_mean=self.pixel_mean,
                                   image_height=self.image_height,
                                   image_width=self.image_width,
                                   num_channels=self.num_channels,
                                   scaling_factor=self.scaling_factor,
                                   use_mean_image=self.use_mean_image)

        # Update instance attributes
        self.model_dict = model_dict

        return model_dict

    ##
    def train_model(self):
        '''
        DataStruct.train_model()

        Returns the trained network and a history of the
        training accuracy/ loss and validation accuracy.
        '''
        # Error check
        if self.model_dict is None:
            print('No models detected, creating default network.')
            model_parameters = ResNetParameters()
            model_dict = models.create(model_parameters,
                                       num_classes=self.num_classes,
                                       pixel_mean=self.pixel_mean,
                                       image_height=self.image_height,
                                       image_width=self.image_width,
                                       num_channels=self.num_channels,
                                       scaling_factor=self.scaling_factor,
                                       use_mean_image=self.use_mean_image)

        else:
            # Check model keys
            valid_model_dict = ['input_var', 'label_var',
                                'net', 'num_classes']
            for elem in self.model_dict.keys():
                if elem not in valid_model_dict:
                    raise TypeError('Invalid model dictionary')

        # Call model training subfunction, allow cross validation
        net, training_hx = models.train(self.model_dict,
                                        reader_train,
                                        train_epoch_size,
                                        learn_params,
                                        self.num_classes,
                                        debug_mode=self.debug_mode,
                                        gpu=self.gpu,
                                        model_save_root=self.model_save_root,
                                        profiler_dir=self.profiler_dir,
                                        reader_test=self.reader_test,
                                        tb_log_dir=self.tb_log_dir,
                                        tb_freq=self.tb_freq,
                                        test_epoch_size=self.test_epoch_size,
                                        extra_aug=self.extra_aug)

        return net, training_hx

    ##
    def extract_features(self):
        '''
        DataStruct.extract_features()

        Returns
        '''
        df = 1
        #TODO return a df with filenames, labels, and features
        #

        return df


    ##
    def evaluate_model(self):
        '''
        DataStruct.evaluate_model()

        Returns a Padas df with performance evaluation on the
        held-out test set.
        '''

        return 1

    ##
    def plot_features(self,
                      df=None,
                      backend='plotly'):
        '''
        DataStruct.plot_features()

        Return a plotting object to visualize the phenotypic
        profiling results using holoviews or Plotly as a backend.

        '''
        # Load sample dataset for debugging
        if self.debug_mode:
            root_dir = os.path.join(os.path.dirname(c_decode.__file__),
                                    '/data/profiling/cells')
            filename = 'Res50_81_cells_mapped.tsv'
            filepath = os.path.join(root_dir, filename)
            df = pd.read_csv(filepath, sep='\t', header=True)
        else:
            raise RuntimeError('This section is not complete!')

        # Call plotting subfunction
        if backend == 'plotly':
            fig = prepare_plotly(df,
                                 title=None,
                                 size=5,
                                 cmap=None,
                                 cscale=None,
                                 opacity=0.9,
                                 mode='markers',
                                 hovermode='closest',
                                 text_labels=None)
        elif backend == 'holoviews':
            raise RuntimeError('This section is not complete!')

        return fig


    ##
    def plot_unique(self, max_size=224):
        '''
        DataStruct.plot_unique()

        Return a Holoviews RGB element containing one
        random unique image per class.
        '''
        hv_img, _ = plot.unique_images(self.mapfile,
                                       text_labels=self.text_labels,
                                       randomize=True,
                                       unique=True,
                                       save_im=True,
                                       max_size=max_size)
        return hv_img

    ##
    def save(self, savepath):
        '''
        DataStruct.save()

        Saves the DataStruct instance to the specified filepath.
        '''

        return 1

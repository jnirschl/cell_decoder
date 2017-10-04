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
import gc

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

# Set root dir
ROOT_DIR = os.path.join(os.path.dirname(c_decoder.__file__))


## Create a data_struct class
class DataStruct:
    '''
    A DataStruct class for storing cell_decoder information to
    train and evaluate cntk models.

    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    # Class attributes
    RESNET_LAYERS = [18, 34, 50, 101, 152]
    IMAGE_READERS = ['default', 'microscopy']
    VALID_MAPFILE_DICT_KEYS = ['train', 'validation', 'test']
    VALID_READER_DICT_KEYS = VALID_MAPFILE_DICT_KEYS
    VALID_MODEL_DICT_KEYS = ['input_var', 'label_var', 'net', 'num_classes']

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
        self.mapfile = mapfile
        self.num_classes = int(df['label'].max() + 1)
        self.mapfile_root = mapfile_root
        self.mapfile_name = mapfile_name
        self.epoch_size = df.shape[0]

    ##
    def read(self):
        '''
        DataStruct.read()

        Returns a the mapfile as a Pandas dataframe
        '''
        #TODO allow reading train/ test mapfile
        #TODO allow reading images based on FP/FN (test set)

        df, _, _ = mapfile_utils.read(self.mapfile)

        return df

    ##
    def compute_image_mean(self,
                           savepath=os.path.join(ROOT_DIR, 'meanfiles'),
                           filename=None,
                           data_aug=True,
                           debug_mode=True,
                           save_img=True,
                           nargout=False):
        '''
        DataStruct.compute_image_mean()

        Computes the mean image and RGB pixel values
        for a given mapfile and saves a PNG and OpenCV
        XML file of the mean image.

        Optional: Returns the mean image as an RGB numpy array.
        '''
        # Set savepath
        if savepath is None:
            savepath = os.path.join(ROOT_DIR, 'meanfiles')
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
                                                filename.replace('.png', '.xml'))

        # Store mean pixel values (BGR)
        n_elem_ch = image_mean[:,:,0].size
        self.pixel_mean = np.mean(image_mean.reshape([n_elem_ch, 3]), axis=0)
        self.pixel_std = np.std(image_mean.reshape([n_elem_ch, 3]), axis=0)

        # Optional output argument
        if nargout:
            # Convert to RGB
            image_mean = cv2.cvtColor(image_mean, cv2.COLOR_BGR2RGB).astype('uint8')

            # Normalize image
#            image_mean = img_utils.to_float(image_mean)

            return image_mean

    ##
    def partition(self,
                  k_fold=5,
                  savepath=os.path.join(ROOT_DIR, 'mapfiles'),
                  held_out_test=0.05,
                  random_seed=None):
        '''
        DataStruct.partition()

        Partitions a mapfile into a training and held-out test
        data set. The training data set is split into k-fold for
        cross-validation.

        '''
        # Set random seed, if not given
        if random_seed is None:
            random_seed = self.random_seed

        # Setup cross validation
        mapfile_dict = mapfile_utils.crossval(mapfile=self.mapfile,
                                              k_fold=k_fold,
                                              held_out_test=held_out_test,
                                              savepath=savepath,
                                              random_seed=random_seed)

        # Assing object attributes
        self.mapfile_dict = mapfile_dict

        return mapfile_dict

    ##
    def create_reader(self,
                      transform_params=None,
                      held_out_test=0.1,
                      random_seed=None,
                      mapfile_dict=None,
                      reader='default',
                      savepath=os.path.join(ROOT_DIR, 'mapfiles'),
                      k_fold=5,
                      allowed_readers=IMAGE_READERS,
                      valid_mapfile_dict=VALID_MAPFILE_DICT_KEYS):
        '''
        DataStruct.create_reader(TransformParams)

        Creates CNTK Minibatch Source readers for training
        and testing mapfiles.
        '''
        if reader not in allowed_readers:
            raise ValueError('Invalid reader {0:s}!'.format(reader))

        # Set default transform parameters, if None
        if transform_params is None:
            print('Using default transform parameters.')
            transform_params = TransformParameters()

        # Override self.mapfile_dict if given as kwarg
        if mapfile_dict is not None:
            print('Input mapfile_dict will override any existing values.')
            self.mapfile_dict = mapfile_dict

        # Create mapfile_dict if none exists
        if self.mapfile_dict is None:
            # TODO update print text for fixed N vs fraction held-out
            print('Partitioning data into training/ validation/' + \
                  'and {0:0.2f}% test datasets.'.format(held_out_test))
            mapfile_dict = partition(k_fold=k_fold,
                                     savepath=savepath,
                                     held_out_test=held_out_test,
                                     random_seed=None)
        else:
            # TODO Check mapfile dict keys
            # Check model keys
            for elem in self.mapfile_dict.keys():
                if elem not in valid_mapfile_dict:
                    raise TypeError('Invalid mapfile dictionary key ({0:s})'.format(elem))

            # If exist, assign to mapfile_dict
            mapfile_dict = self.mapfile_dict


        # Check length of train/ test fold
        train_fold = len(mapfile_dict['train'])
        validation_fold = len(mapfile_dict['validation'])
        assert (train_fold == validation_fold), \
            'Training ({0:d}) and cross-validation ({1:d}) data partitions' +\
            'must have the same number of folds!'.format(train_fold,
                                                         validation_fold)

        # TODO allow no validation dict

        # Create minibatch sources from mapfile_dict
        train_list = []
        valid_list = []
        for fold, (train_map, valid_map) in enumerate(zip(mapfile_dict['train'],
                                                          mapfile_dict['validation'])):
            if reader.lower() == 'default':
                train_reader = default_reader.image(train_map,
                                                    transform_params,
                                                    num_classes=self.num_classes,
                                                    num_channels=self.num_channels,
                                                    height=self.image_height,
                                                    width=self.image_width,
                                                    mean_filepath=self.image_mean_filepath,
                                                    is_training=True,
                                                    use_mean_image=self.use_mean_image,
                                                    verbose=(fold == 0))

                valid_reader = default_reader.image(valid_map,
                                                    transform_params,
                                                    num_classes=self.num_classes,
                                                    num_channels=self.num_channels,
                                                    height=self.image_height,
                                                    width=self.image_width,
                                                    mean_filepath=self.image_mean_filepath,
                                                    is_training=False,
                                                    use_mean_image=self.use_mean_image,
                                                    verbose=False)


                # Append to list
                train_list.append(train_reader)
                valid_list.append(valid_reader)

            elif reader.lower()=='microscopy':
                raise RuntimeError('This section not complete!')

        # Create held-out-test reader, if available
        if mapfile_dict['test'] is not None:
            print('\nCreating test reader')

            # Create output list
            test_list = []
            for test_map in mapfile_dict['test']:
                test_reader = default_reader.image(test_map,
                                                   transform_params,
                                                   num_classes=self.num_classes,
                                                   num_channels=self.num_channels,
                                                   height=self.image_height,
                                                   width=self.image_width,
                                                   mean_filepath=self.image_mean_filepath,
                                                   is_training=False,
                                                   use_mean_image=self.use_mean_image,
                                                   verbose=False)
                # Append list
                test_list.append(test_reader)

        # Assign reader_dict
        if mapfile_dict['test'] is None:
            reader_dict = {'train':train_list,
                           'validation':valid_list,
                           'test':[None]}
        else:
            reader_dict = {'train':train_list,
                           'validation':valid_list,
                           'test':test_list}

        # Assing object attributes
        self.reader_dict = reader_dict

        return reader_dict

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
                    raise TypeError('Invalid model parameter key ({0:s})'.format(elem))

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
    def train_model(self,
                    debug_mode=False,
                    model_dict=None,
                    reader_dict=None,
                    learn_params=None,
                    max_epochs=100,
                    mb_size=64,
                    valid_model_dict=VALID_MODEL_DICT_KEYS,
                    valid_reader_dict=VALID_READER_DICT_KEYS,
                    verbose=True):
        '''
        DataStruct.train_model()

        Returns the trained network and a history of the
        training accuracy/ loss and validation accuracy.
        '''
        if debug_mode or self.debug_mode:
            max_epochs = 2
            debug_mode = True

        # Override self.reader_dict if given as kwarg
        if reader_dict is not None:
            print('Input reader_dict will override any existing values.')
            self.reader_dict = reader_dict

        # Create mapfile_dict if none exists
        if self.reader_dict is None:
            print('No readers detected, creating default reader.')
            reader_dict = create_reader(held_out_test=0.1,
                                        k_fold=5,
                                        mapfile_dict=None,
                                        random_seed=None,
                                        reader='default',
                                        savepath=os.path.join(ROOT_DIR, 'mapfiles'),
                                        transform_params=TransformParameters())

        else:
            # Check model keys
            for elem in self.reader_dict.keys():
                if elem not in valid_reader_dict:
                    raise TypeError('Invalid reader dictionary key ({0:s})'.format(elem))

            # If exists, assign to reader_dict
            reader_dict = self.reader_dict

        # Override self.model_dict if given as kwarg
        if model_dict is not None:
            print('Input model_dict will override any existing values.')
            self.model_dict = model_dict

        # Create model_dict if none exists
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
            for elem in self.model_dict.keys():
                if elem not in valid_model_dict:
                    raise TypeError('Invalid model dictionary key ({0:s})'.format(elem))

            model_dict = self.model_dict

        # Create and compile learn_params if none exists
        if learn_params is None:
            print('Using default learning parameters.')

            learn_params = []
            for fold in  reader_dict['train']:
                train_learn = LearningParameters(max_epochs=max_epochs,
                                                 mb_size=mb_size)
                learn_dict = train_learn.compile(model_dict, fold['epoch_size'])
                learn_params.append(learn_dict)

        # Allocate output vars
        training_summary = {'model_path':[],
                            'train_df':[] }

        # Train models using cross-validation
        print(20*'-')
        for fold, (r_train, r_valid) in enumerate(zip(reader_dict['train'],
                                                      reader_dict['validation'])):
            if verbose:
                print('Fold:\t{0:02d}'.format(fold))

            # Call model training subfunction
            model_path, train_df = models.train(model_dict,
                                                r_train['mb_source'],
                                                r_train['epoch_size'],
                                                learn_params[fold],
                                                reader_valid=r_valid['mb_source'],
                                                valid_epoch_size=r_valid['epoch_size'],
                                                debug_mode=debug_mode,
                                                gpu=self.gpu,
                                                model_save_root=self.model_save_root,
                                                profiler_dir=self.profiler_dir,
                                                rgb_mean=np.median(self.pixel_mean),
                                                rgb_std=np.median(self.pixel_std),
                                                tb_log_dir=self.tb_log_dir,
                                                tb_freq=self.tb_freq,
                                                extra_aug=True)

            # Clear gpu memory
            gc.collect()

            # Save to output dictionary
            training_summary['model_path'].append(model_path)
            training_summary['train_df'].append(train_df)

            print(20*'-' + '\n')

        print('Training complete!')

        # Add column for training fold and append dataframes
        output = pd.DataFrame()
        for fold, df in enumerate(training_summary['train_df']):
            df = df.join(pd.DataFrame(fold*np.ones((df.shape[0],1), dtype=int),
                                      columns={'fold'}) )
            output = output.append(df) #  ignore_index=True


        training_summary['train_df'] = output #[['sample_count', 'mb_index',
#                                               'epoch_index', 'train_loss', 'train_error',
#                                               'test_error']]

        return training_summary

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
            ROOT_DIR = os.path.join(os.path.dirname(c_decode.__file__),
                                    '/data/profiling/cells')
            filename = 'Res50_81_cells_mapped.tsv'
            filepath = os.path.join(ROOT_DIR, filename)
            df = pd.read_csv(filepath, sep='\t', header=True)
        else:
            raise NotImplemented('This section is not complete!')

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
            raise NotImplemented('This section is not complete!')

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
    def get_next_minibatch(self,
                           partition='train',
                           fold=0,
                           num_samples=1):
        '''
        DataStruct.get_next_minibatch
        '''
        # Get reader_dict
        if self.reader_dict is None:
            reader_dict = create_reader(held_out_test=0.1,
                                        k_fold=5,
                                        mapfile_dict=None,
                                        random_seed=None,
                                        reader='default',
                                        savepath=os.path.join(ROOT_DIR, 'mapfiles'),
                                        transform_params=TransformParameters())
            self.reader_dict = reader_dict
        else:
            reader_dict = self.reader_dict

        # Get model_dict
        if self.model_dict is None:
            raise RuntimeError('Please run create_model first!')
        else:
            input_var = self.model_dict['input_var']
            label_var = self.model_dict['label_var']

        reader_train =  reader_dict[partition][fold]['mb_source']
        input_map = {
            'features': reader_train.streams.features,
            'labels': reader_train.streams.labels
        }

        data = reader_train.next_minibatch(num_samples,
                                           input_map=input_map)

        return data

    ##
    def save(self,
             savepath=ROOT_DIR):
        '''
        DataStruct.save()

        Saves the DataStruct instance to the specified filepath.
        '''

        return 1

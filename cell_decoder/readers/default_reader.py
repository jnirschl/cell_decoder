# cell_decoder/readers/default_reader.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ==============================================================================
'''

.. module::cell_decoder.readers.default_reader
    :synopsis: Default image reader using ImageDeserializer

Copyright (c) 2017 Jeffrey J. Nirschl
'''

# Standard library imports
import os
import numpy as np

# CNTK imports
import cntk.io.transforms as xforms
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs, INFINITELY_REPEAT

# cell_decoder imports
from cell_decoder.config import TransformParameters
from cell_decoder.io import mapfile_utils

##
def image(mapfile,
          transform_params,
          num_classes=None,
          num_channels=3,
          height=224,
          width=224,
          mean_filepath=None,
          is_training=True,
          use_mean_image=True,
          verbose=True):
    '''
    default_reader.image(mapfile, transform_params, num_classes,
                        is_training=True, use_image_mean=True)
    '''
    if not (type(transform_params) == type(TransformParameters())):
        raise ValueError('transform_params must be a valid TransformParameters class.')

    # Allocate empty transforms list
    transforms = []

    # Read mapfile and set epoch_size
    df, _, _ = mapfile_utils.read(mapfile)
    epoch_size = df.shape[0]
    
    # Set num_classes, if not entered
    if num_classes is None:
        num_classes = len(df['labels'].unique())

    # Random cropping and side_ratio jitter during training
    if is_training:
        transforms += [
            xforms.crop(crop_type = transform_params.crop_type,
                        crop_size = transform_params.crop_size,
                        side_ratio = transform_params.side_ratio,
                        area_ratio = transform_params.area_ratio,
                        aspect_ratio = transform_params.aspect_ratio,
                        jitter_type = transform_params.jitter_type),
            xforms.color(brightness_radius = transform_params.brightness_radius,
                         contrast_radius = transform_params.contrast_radius,
                         saturation_radius = transform_params.saturation_radius)
        ]
    else:
        transforms += [
            xforms.crop(crop_type = transform_params.crop_type,
                        side_ratio = transform_params.side_ratio)
        ]

    # Scaling and mean subtraction
    if use_mean_image and mean_filepath:
        assert(os.path.isfile(mean_filepath))
        # Subtract mean image
        transforms += [
            xforms.scale(width,
                         height,
                         num_channels,
                         interpolations=transform_params.interpolations),
            xforms.mean(mean_filepath)
        ]

        if verbose:
            print('Subtracting mean image.')
    else:
        # Subtract pixel mean
        transforms += [
            xforms.scale(width,
                         height,
                         num_channels,
                         interpolations=transform_params.interpolations)
        ]

        if verbose:
            print('Subtracting mean pixel values.')

    # Initialize image deserializer
    deserializer = ImageDeserializer(filename=mapfile,
                                     streams=StreamDefs(
                                         features=StreamDef(field='image', transforms=transforms),
                                         labels=StreamDef(field='label', shape=num_classes))
    )

    # Create minibatch source
    mb_source  = MinibatchSource(deserializers=deserializer,
                                 randomize=is_training,
                                 max_sweeps=INFINITELY_REPEAT if is_training else 1)

    return {'mb_source':mb_source,
            'epoch_size':epoch_size}

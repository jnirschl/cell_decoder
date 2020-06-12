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

.. module:cell_decoder.utils.python_layer
    :synopsis: Python layer for applying transforms to a minibatch.

'''

# Imports
import numpy as np

from cntk.io import MinibatchData
from cntk.core import NDArrayView, Value
from cntk.device import gpu

from cell_decoder.img import img_utils


##
def process_minibatch(minibatch_dict,
                      input_var,
                      label_var,
                      deterministic=False,
                      do_rot=True,
                      do_flip=True,
                      swap_ch='rg',
                      drop_ch='rg',
                      gauss_blur=True,
                      gauss_noise=True,
                      im_height=224,
                      im_width=224,
                      n_ch=3,
                      rgb_mean=25,
                      rgb_std=5,
                      device=gpu(0)):
    '''
    python_layer.process_minibatch
    '''
    assert isinstance(minibatch_dict, dict), 'Input must be class: MinibatchData'

    # Extract streams
    features = minibatch_dict[input_var].data
    labels =  minibatch_dict[label_var].data
    mb_size = minibatch_dict[label_var].num_samples

    # Pre-allocate the random vars for transforms
    transform_dict = set_transforms(mb_size,
                                    deterministic=deterministic,
                                    do_flip=do_flip,
                                    do_rot=do_rot,
                                    gauss_blur=gauss_blur,
                                    gauss_noise=gauss_noise,
                                    swap_ch=swap_ch,
                                    drop_ch=drop_ch)

    # Apply tranforms
    features = transform_layer(features,
                               labels,
                               transform_dict,
                               device=device,
                               im_height=im_height,
                               im_width=im_width,
                               n_ch=n_ch)

    # Re-assign to MinibatchData
    minibatch_dict[input_var].data = features

    return minibatch_dict

##
def transform_layer(features,
                    labels,
                    transform_dict,
                    im_height=224,
                    im_width=224,
                    n_ch=3,
                    device=gpu(0)):
    '''

    '''
    # Convert mb_data object: Value(NDarrayView) into numpy ndarray
    features = features.data.asarray()
    labels = labels.data.asarray()

    # For loop over images in minibatch
    # Each element is Format is:
    #     [N, 1, C, H, W]
    #     Channels are in bgr format
    for idx in range(features.shape[0]):
        # Extract 1 img and convert to HWC
        Orig_im = features[idx][0].swapaxes(0, 2)

        # Apply transforms
        T_im = img_utils.apply_transform(Orig_im,
                                         np.argmax(labels[idx]),
                                         transform_dict['do_flip'][idx],
                                         transform_dict['do_rot'][idx],
                                         transform_dict['drop_ch'],
                                         transform_dict['flip_type'][idx],
                                         transform_dict['gauss_blur'][idx],
                                         transform_dict['gauss_noise'][idx],
                                         transform_dict['rot_deg'][idx],
                                         transform_dict['swap_ch'],
                                         dtype=Orig_im.dtype,
                                         gauss_sigma=transform_dict['gauss_std'][idx],
                                         im_height=im_height,
                                         im_width=im_width,
                                         n_ch=n_ch,
                                         stain_aug=transform_dict['stain_aug'])

        # Return to [N, 0, C, H, W]
        features[idx][0] = T_im.swapaxes(0, 2)

    # Convert numpy features into a cntk Value(NDArrayView)
    features = Value(NDArrayView.from_dense(features, device=device),
                     device=device)

    return features

##
def set_transforms(mb_size,
                   deterministic=False,
                   do_flip=True,
                   do_rot=True,
                   drop_ch='rg',
                   gauss_blur=True,
                   gauss_noise=True,
                   swap_ch='rg',
                   stain_aug=True):
    '''
    python_layer.set_transforms(mb_size)

    Returns of dictionary of transforms, random set
    for each image in a minibatch.
    '''
    # Flip switch (bernoulli, p = 0.5)
    # Flip type (ud vs lr vs ud + lr, multinoulli)
    if deterministic or (not do_flip):
        do_flip  = np.zeros((mb_size, 1), dtype=bool)
        flip_type = np.random.randint(2, size=(mb_size, 1),
                                      dtype=bool)
    else:
        do_flip = np.random.randint(2, size=(mb_size, 1), dtype=bool)
        flip_type = np.random.randint(-1,2, size=(mb_size, 1), dtype=int)

    # Rotation switch (bernoulli, p = 0.5)
    if deterministic or (not do_rot):
        do_rot = np.zeros((mb_size, 1), dtype=bool)
        rot_deg = np.zeros((mb_size, 1), dtype=int)
    else:
        do_rot = np.random.randint(2, size=(mb_size, 1), dtype=bool)
        rot_deg = np.random.randint(1,high=5, size=(mb_size, 1), dtype=int)

    # Gaussian blur switch
    if deterministic or (not gauss_blur):
        gauss_blur = np.zeros((mb_size, 1), dtype=bool)
        gauss_std = np.zeros((mb_size, 1), dtype=int)
    else:
        gauss_blur = np.random.randint(2, size=(mb_size, 1), dtype=bool)
        gauss_std = np.clip(np.random.randn(mb_size, 1)*.5 + 1, a_min=0.25, a_max=3)

    # Gaussian noise switch
    if deterministic or (not gauss_noise):
        gauss_noise = np.zeros((mb_size, 1), dtype=bool)
    else:
        gauss_noise = np.random.randint(2, size=(mb_size, 1), dtype=bool)

    # Assign output dict
    transform_dict = {'do_rot':do_rot, 'rot_deg':rot_deg,
                      'do_flip':do_flip, 'flip_type':flip_type,
                      'gauss_blur':gauss_blur, 'gauss_std':gauss_std,
                      'gauss_noise':gauss_noise, 'swap_ch':swap_ch,
                      'drop_ch':drop_ch, 'stain_aug':stain_aug
    }

    return transform_dict

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
from cntk.io import MinibatchData
from cntk.core import NDArrayView, Value
from cntk.device import gpu

from cell_decoder.img_utils import apply_transform



##
def process_minibatch(mb_dict, deterministic=False, device=gpu(0)):
    '''
    python_layer.process_minibatch
    '''
    assert isinstance(mb_dict, dict), 'Input must be a MinibatchData class object'

    # Extract streams
    features = minibatch_dict['features'].data
    label =  minibatch_dict['labels'].data

    # Pre-allocate the random vars for transforms
    transform_opts = set_transforms(mb_size, deterministic=deterministic):

    # Apply tranforms
    features = apply_transform(features,
                               label,
                               do_rot,
                               rot_deg,
                               do_flip,
                               flip_type,
                               swap_ch,
                               drop_ch,
                               gauss_blur=gauss_blur,
                               gauss_sigma=gauss_sigma,
                               gauss_noise=gauss_noise,
                               transform_opts=transform_opts)

    # Convert numpy features into a cntk Value(NDArrayView)
    features = Value(NDArrayView.from_dense(features, device=device),
                     device=device)

    # Re-assign to MinibatchData
    minibatch_dict['features'].data = features

    return minibatch_dict

##
def transform_layer(features,
                    labels,
                    do_rot,
                    rot_deg,
                    do_flip,
                    flip_type,
                    swap_ch,
                    drop_ch,
                    gauss_blur,
                    gauss_noise):
    '''

    '''
    # For loop over images in minibatch
    # Each element is Format is:
    #     [N, 1, C, H, W]
    #     Channels are in bgr format
    for idx in range(0, features.shape[0]):
        # Extract 1 img and convert to HWC
        T_im = features[idx][0].swapaxes(0, 2)

        # Apply transforms
        T_im = apply_transforms(T_img,
                                do_rot,
                                rot_deg,
                                do_flip,
                                flip_type,
                                swap_ch,
                                drop_ch,
                                gauss_blur=gauss_blur,
                                gauss_noise=gauss_noise,
                                gauss_kernel=gauss_kernel,
                                gauss_sigma=gauss_sigma) # label[idx]

        # Return to [N, 0, C, H, W]
        features[idx][0] = T_im.swapaxes(0, 2)

    return features

##
def set_transforms(mb_size,
                   deterministic=False):
    '''
    '''
    # Flip switch (bernoulli, p = 0.5)
    # Flip type (ud vs lr vs ud + lr, multinoulli)
    if deterministic:
        do_flip  = np.zeros((mb_size, 1), dtype=bool)
        flip_type = np.random.randint(2, size=(mb_size, 1),
                                      dtype=bool)
    else:
        do_flip = np.random.randint(2, size=(mb_size, 1), dtype=bool)
        flip_type = np.random.randint(-1,2, size=(mb_size, 1), dtype=int)

    # Rotation switch (bernoulli, p = 0.5)
    if deterministic:
        do_rot = np.zeros((mb_size, 1), dtype=bool)
        rot_deg = np.zeros((mb_size, 1), dtype=int)
    else:
        do_rot = np.random.randint(2, size=(mb_size, 1), dtype=bool)
        rot_deg = np.random.randint(1,high=5, size=(mb_size, 1), dtype=int)

    # Gaussian blur switch
    if deterministic:
        gauss_blur = np.zeros((mb_size, 1), dtype=bool)
        gauss_std = np.zeros((mb_size, 1), dtype=int)
        gauss_noise = np.zeros((mb_size, 1), dtype=bool)
        #         gau = np.zeros((mb_size, 1), dtype=int)
    else:
        gauss_blur = np.random.randint(2, size=(mb_size, 1), dtype=bool)
        gauss_std = np.clip(np.random.randn(mb_size, 1)*1 + 3, a_min=0.5, a_max=9)


    return do_rot, rot_deg, do_flip, flip_type, gauss_blur, gauss_std, gauss_noise

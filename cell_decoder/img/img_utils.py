# cell_decoder/img/img_utils.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ======================================================================
'''
Cell DECODER function to compute the mean image.

Copyright (c) 2017 Jeffrey J. Nirschl
'''

# Standard library imports
import os

# Image and array operations
import cv2
import numpy as np
import pandas as pd

# Imports for computing the image mean and writing to xml


# Third party
from cell_decoder import utils

# Specify opencv optimization
cv2.setUseOptimized(True)


##
def to_float(img):
    #TODO update
    im32f = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return im32f


##
def apply_transform(T_img,
                    do_rot,
                    rot_deg,
                    do_flip,
                    flip_type,
                    swap_ch,
                    drop_ch,
                    gauss_blur=True,
                    gauss_noise=False,
                    gauss_kernel=(9, 9),
                    gauss_sigma=3,
                    label=1,
                    dtype=np.uint8,
                    transform_opts=None,
                    valid_labels=[1]):
    '''
    img_utils.apply_transorm()

    '''
    # Set transform opts if none given
    if transform_opts:
        do_rot = transform_opts
        rot_deg = transform_opts
        do_flip = transform_opts
        flip_type = transform_opts
        swap_ch = transform_opts
        drop_ch = transform_opts
        gauss_blur=True,
        gauss_noise=False,
        gauss_kernel=(9, 9),
        gauss_sigma=3,
    
    # Rotation
    if do_rot and rot_deg != 4:
        T_im = np.rot90(T_im, rot_deg)

    # Flip image vertical/ horizontal: 0 = ud; >0 = lr; <0 = ud + lr
    if do_flip:
        T_im = cv2.flip(T_im, flip_type)

    # Randomly swap channels
    if swap_ch and label in []:
        match_idx, diff_idx = utils.string_cmp(swap_ch.lower(), 'bgr',
                                               match_case=False)
        swap_idx = np.insert(np.random.permutation(match_idx),
                             diff_idx, diff_idx)
        T_im = T_im[:,:,swap_idx]

    # Randomly drop one or more channels given a specified string
    #  -- opencv stores images as bgr --
    if drop_ch:
        match_idx, diff_idx = utils.string_cmp(drop_ch, 'bgr',
                                               match_case=False)

        drop_num = np.random.randint(len(match_idx)+1)

        drop_idx = np.random.choice(match_idx,
                                    size=drop_num,
                                    replace=False)
        for idx in drop_idx:
            T_im[:,:,idx] = np.zeros_like((T_img[:,:,0]))

    # Apply low-pass gaussian filter
    if gauss_blur:
        T_im = cv2.GaussianBlur(T_im, gauss_kernel, gauss_sigma)

    # Additive Gaussian noise
    if gauss_noise:
        I_std = 5 # fluorescence microscopy data
        I_mu = 30 # fluorescence microscopy data
        rnd = np.clip(np.random.randn( T_im.shape )*I_std + I_mu,
                      a_min=0, a_max=255)
        T_im = cv2.add(T_im, rnd.astype(dtype))

    # Histology stain augmentation based on https://arxiv.org/pdf/1707.06183.pdf
    if stain_aug:
        var_a = np.sort(np.array([0.9, 1.1]))
        var_c = np.sort(np.array([-10, 10]))

        T_im  = np.clip(np.add(np.multiply(T_im, np.random.uniform(var_a[0], var_a[1], T_im.shape[-1])),
                               np.random.uniform(var_c[0], var_c[1], T_im.shape[-1])),
                        a_min=0, a_max=255).astype(dtype)

    return T_im

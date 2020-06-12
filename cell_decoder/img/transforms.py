# transforms.py -- a module within img_utils
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Adapted from Microsoft FERPlus/src/img_util.py
# MIT license
#
# Distributable under a BSD "two-clause" license.
# ==============================================================================

# Imports
import numpy as np
import random as rnd
import cv2
from scipy import ndimage
#from rect_util import Rect # Source from FERPl TODO

# Specify opencv optimization
cv2.setUseOptimized(True)

## Rotate image
# ----- Adapted from PyImageSearch imutils/convenience.py
# ----- MIT license
def rotate(img, angle, center=None, scale=1.0):
    '''rotate
    ----- Adapted from PyImageSearch imutils/convenience.py -----
    ----- MIT license -----
    '''
    # grab the dimensions of the image
    (img_height, img_width) = img.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (img_width // 2, img_height // 2)

    # perform the rotation
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    # return the rotated image
    return cv2.warpAffine(img, rot_mat, (img_width, img_height))

## Resize image, preserving aspect ratio
# ----- Adapted from PyImageSearch imutils/convenience.py
# ----- MIT license
def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    '''resize
    ----- Adapted from PyImageSearch imutils/convenience.py -----
    ----- MIT license -----
    '''
    # initialize the dimensions of the image to be resized and
    # grab the image size
    out_dim = None
    (h, w) = img.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        out_dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        out_dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(img, out_dim, interpolation=inter)

    # return the resized image
    return resized, out_dim

# Compute normalization matrix
# ----- Adapted from Microsoft FERPlus/src/img_util.py -----
# ----- MIT license
def compute_norm_mat(base_width, base_height): 
    # normalization matrix used in image pre-processing 
    x      = np.arange(base_width)
    y      = np.arange(base_height)
    X, Y   = np.meshgrid(x, y)
    X      = X.flatten()
    Y      = Y.flatten() 
    A      = np.array([X*0+1, X, Y]).T 
    A_pinv = np.linalg.pinv(A)
    return A, A_pinv

# Preprocess image
# ----- Adapted from Microsoft FERPlus/src/img_util.py -----
# ----- MIT license
def preproc_img(img, A, A_pinv):
    # compute image histogram 
    img_flat = img.flatten()
    img_hist = np.bincount(img_flat, minlength = 256)

    # cumulative distribution function 
    cdf = img_hist.cumsum() 
    cdf = cdf * (2.0 / cdf[-1]) - 1.0 # normalize 

    # histogram equalization 
    img_eq = cdf[img_flat] 

    diff = img_eq - np.dot(A, np.dot(A_pinv, img_eq))

    # after plane fitting, the mean of diff is already 0 
    std = np.sqrt(np.dot(diff,diff)/diff.size)
    if std > 1e-6: 
        diff = diff/std
    return diff.reshape(img.shape)


def distort_img(img, roi, out_width, out_height, max_shift, max_scale, max_angle, max_skew, flip=True): 
    shift_y = out_height*max_shift*rnd.uniform(-1.0,1.0)
    shift_x = out_width*max_shift*rnd.uniform(-1.0,1.0)

    # rotation angle 
    angle = max_angle*rnd.uniform(-1.0,1.0)

    #skew 
    sk_y = max_skew*rnd.uniform(-1.0, 1.0)
    sk_x = max_skew*rnd.uniform(-1.0, 1.0)

    # scale 
    scale_y = rnd.uniform(1.0, max_scale) 
    if rnd.choice([True, False]): 
        scale_y = 1.0/scale_y 
    scale_x = rnd.uniform(1.0, max_scale) 
    if rnd.choice([True, False]): 
        scale_x = 1.0/scale_x 
    T_im = crop_img(img, roi, out_width, out_height, shift_x, shift_y, scale_x, scale_y, angle, sk_x, sk_y)
    if flip and rnd.choice([True, False]): 
        T_im = np.fliplr(T_im)
    return T_im


def crop_img(img, roi, crop_width, crop_height, shift_x, shift_y, scale_x, scale_y, angle, skew_x, skew_y):
    # current face center 
    ctr_in = np.array((roi.center().y, roi.center().x))
    ctr_out = np.array((crop_height/2.0+shift_y, crop_width/2.0+shift_x))
    out_shape = (crop_height, crop_width)
    s_y = scale_y*(roi.height()-1)*1.0/(crop_height-1)
    s_x = scale_x*(roi.width()-1)*1.0/(crop_width-1)
    
    # rotation and scale 
    ang = angle*np.pi/180.0 
    transform = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    transform = transform.dot(np.array([[1.0, skew_y], [0.0, 1.0]]))
    transform = transform.dot(np.array([[1.0, 0.0], [skew_x, 1.0]]))
    transform = transform.dot(np.diag([s_y, s_x]))
    offset = ctr_in-ctr_out.dot(transform)

    # each point p in the output image is transformed to pT+s, where T is the matrix and s is the offset
    T_im = ndimage.interpolation.affine_transform(input = img, 
                                                  matrix = np.transpose(transform), 
                                                  offset = offset, 
                                                  output_shape = out_shape, 
                                                  order = 1,   # bilinear interpolation 
                                                  mode = 'reflect', 
                                                  prefilter = False)
    return T_im




# Random crop with side ratio
# Contrast/ color adjustment
# Jitter
# Rotate
# Interchage channels
def crop():
     '''img_utils.crop()
     
     Accepts X and returns y.
     Copyright (c) 2017 Jeffrey J. Nirschl
     '''
     # Random crop

     # Crop to center (padding?)
     return img

def color_normalize(img):
     '''img_utils.preprocess
     Accepts X and returns y.
     Copyright (c) 2017 Jeffrey J. Nirschl
     '''
     return img

def preprocess(img):
     '''img_utils.preprocess
     Accepts X and returns y.
     Copyright (c) 2017 Jeffrey J. Nirschl
     '''
     
     return img

def scale(img):
     '''img_utils.preprocess
     Accepts X and returns y.
     Copyright (c) 2017 Jeffrey J. Nirschl
     '''
     if rnd.choice([True, False]): 
       scale_y = 1.0/scale_y 
     return img


def affine_transforms(img):
     '''img_utils.preprocess
     Accepts X and returns y.
     Copyright (c) 2017 Jeffrey J. Nirschl
     '''
     # Scale
     if rnd.choice([True, False]): 
          scale_y = 1.0/scale_y
          
     # Reflect
     if rnd.choice([True, False]): 
          scale_y = 1.0/scale_y
          
     # Rotate
     if rnd.choice([True, False]): 
          scale_y = 1.0/scale_y
          
     # Shear
     if rnd.choice([True, False]): 
          scale_y = 1.0/scale_y
     

     return img
     
def filter(img):
    '''img_utils.preprocess
    Accepts X and returns y.
    Copyright (c) 2017 Jeffrey J. Nirschl
    '''
    return img

def apply_transform(img, roi, out_width, out_height,
                    max_shift, max_scale, max_angle, max_skew,
                    rotateflip=True):
     

    # Random crop with side ratio
    # Contrast/ color adjustment
    # Jitter 
    # Rotate
    # Interchage channels
    shift_y = out_height*max_shift*rnd.uniform(-1.0,1.0)
    shift_x = out_width*max_shift*rnd.uniform(-1.0,1.0)

    # rotation angle 
    angle = max_angle*rnd.uniform(-1.0,1.0)

    #skew 
    sk_y = max_skew*rnd.uniform(-1.0, 1.0)
    sk_x = max_skew*rnd.uniform(-1.0, 1.0)

    # scale 
    scale_y = rnd.uniform(1.0, max_scale) 
    if rnd.choice([True, False]): 
        scale_y = 1.0/scale_y 
    scale_x = rnd.uniform(1.0, max_scale)
    if rnd.choice([True, False]): 
        scale_x = 1.0/scale_x

    # Crop image
    
    T_im = crop_img(img, roi, out_width, out_height, shift_x, shift_y, scale_x, scale_y, angle, sk_x, sk_y)
    if flip and rnd.choice([True, False]): 
        T_im = np.fliplr(T_im)
    return T_im

def crop_img(img, roi, crop_width, crop_height, shift_x, shift_y, scale_x, scale_y, angle, skew_x, skew_y):
    # current face center 
    ctr_in = np.array((roi.center().y, roi.center().x))
    ctr_out = np.array((crop_height/2.0+shift_y, crop_width/2.0+shift_x))
    out_shape = (crop_height, crop_width)
    s_y = scale_y*(roi.height()-1)*1.0/(crop_height-1)
    s_x = scale_x*(roi.width()-1)*1.0/(crop_width-1)
    
    # rotation and scale 
    ang = angle*np.pi/180.0 
    transform = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    transform = transform.dot(np.array([[1.0, skew_y], [0.0, 1.0]]))
    transform = transform.dot(np.array([[1.0, 0.0], [skew_x, 1.0]]))
    transform = transform.dot(np.diag([s_y, s_x]))
    offset = ctr_in-ctr_out.dot(transform)

    # each point p in the output image is transformed to pT+s, where T is the matrix and s is the offset
    T_im = ndimage.interpolation.affine_transform(input = img, 
                                                  matrix = np.transpose(transform), 
                                                  offset = offset, 
                                                  output_shape = out_shape, 
                                                  order = 1,   # bilinear interpolation 
                                                  mode = 'reflect', 
                                                  prefilter = False)
    return T_im


#import numpy as np
#import random as rnd
#from PIL import Image
#from scipy import ndimage
#from rect_util import Rect
#
#def compute_norm_mat(base_width, base_height): 
#    # normalization matrix used in image pre-processing 
#    x      = np.arange(base_width)
#    y      = np.arange(base_height)
#    X, Y   = np.meshgrid(x, y)
#    X      = X.flatten()
#    Y      = Y.flatten() 
#    A      = np.array([X*0+1, X, Y]).T 
#    A_pinv = np.linalg.pinv(A)
#    return A, A_pinv
#
#def preproc_img(img, A, A_pinv):
#    # compute image histogram 
#    img_flat = img.flatten()
#    img_hist = np.bincount(img_flat, minlength = 256)
#
#    # cumulative distribution function 
#    cdf = img_hist.cumsum() 
#    cdf = cdf * (2.0 / cdf[-1]) - 1.0 # normalize 
#
#    # histogram equalization 
#    img_eq = cdf[img_flat] 
#
#    diff = img_eq - np.dot(A, np.dot(A_pinv, img_eq))
#
#    # after plane fitting, the mean of diff is already 0 
#    std = np.sqrt(np.dot(diff,diff)/diff.size)
#    if std > 1e-6: 
#        diff = diff/std
#    return diff.reshape(img.shape)
#
#def distort_img(img, roi, out_width, out_height, max_shift, max_scale, max_angle, max_skew, flip=True): 
#    shift_y = out_height*max_shift*rnd.uniform(-1.0,1.0)
#    shift_x = out_width*max_shift*rnd.uniform(-1.0,1.0)
#
#    # rotation angle 
#    angle = max_angle*rnd.uniform(-1.0,1.0)
#
#    #skew 
#    sk_y = max_skew*rnd.uniform(-1.0, 1.0)
#    sk_x = max_skew*rnd.uniform(-1.0, 1.0)
#
#    # scale 
#    scale_y = rnd.uniform(1.0, max_scale) 
#    if rnd.choice([True, False]): 
#        scale_y = 1.0/scale_y 
#    scale_x = rnd.uniform(1.0, max_scale) 
#    if rnd.choice([True, False]): 
#        scale_x = 1.0/scale_x 
#    T_im = crop_img(img, roi, out_width, out_height, shift_x, shift_y, scale_x, scale_y, angle, sk_x, sk_y)
#    if flip and rnd.choice([True, False]): 
#        T_im = np.fliplr(T_im)
#    return T_im
#
#def crop_img(img, roi, crop_width, crop_height, shift_x, shift_y, scale_x, scale_y, angle, skew_x, skew_y):
#    # current face center 
#    ctr_in = np.array((roi.center().y, roi.center().x))
#    ctr_out = np.array((crop_height/2.0+shift_y, crop_width/2.0+shift_x))
#    out_shape = (crop_height, crop_width)
#    s_y = scale_y*(roi.height()-1)*1.0/(crop_height-1)
#    s_x = scale_x*(roi.width()-1)*1.0/(crop_width-1)
#    
#    # rotation and scale 
#    ang = angle*np.pi/180.0 
#    transform = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
#    transform = transform.dot(np.array([[1.0, skew_y], [0.0, 1.0]]))
#    transform = transform.dot(np.array([[1.0, 0.0], [skew_x, 1.0]]))
#    transform = transform.dot(np.diag([s_y, s_x]))
#    offset = ctr_in-ctr_out.dot(transform)
#
#    # each point p in the output image is transformed to pT+s, where T is the matrix and s is the offset
#    T_im = ndimage.interpolation.affine_transform(input = img, 
#                                                  matrix = np.transpose(transform), 
#                                                  offset = offset, 
#                                                  output_shape = out_shape, 
#                                                  order = 1,   # bilinear interpolation 
#                                                  mode = 'reflect', 
#                                                  prefilter = False)
#    return T_im

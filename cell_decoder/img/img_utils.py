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


# Specify opencv optimization
cv2.setUseOptimized(True)


##
def to_float(img):
    im32f = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return im32f

## 

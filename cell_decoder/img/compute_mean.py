# cell_decoder/img/compute_image_mean.py
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
from io import BytesIO
from lxml import etree

# Third party
from cell_decoder.io import mapfile_utils

# Specify opencv optimization
cv2.setUseOptimized(True)

## TODO save RGB intensity eigenvalue and eigenvector
# https://github.com/Microsoft/CNTK/blob/master/Examples/Image/DataSets/ImageNet/ImageNet1K_intensity.xml
# Also see:
# https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image

##
# PCA augmentation
# The second form of data augmentation consists of altering the intensities of the RGB channels in training images. Specifically, we perform PCA on the set of RGB pixel values throughout the ImageNet training set. To each training image, we add multiples of the found principal components, with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1.
#
# https://deshanadesai.github.io/notes/Fancy-PCA-with-Scikit-Image
# 1) Read data and compute mean
# 2) Subtract mean
# 3) Compute the cov mat
# 4) Calculate the Eigenvectors and eigenvalues of the cov mat

## TODO - update
def image(mapfile,
          df=None,
          savepath=None,
          filename="mean_img.png",
          data_aug=True,
          debug_mode=True,
          save_img=True):
    '''compute_mean.image()

    Accepts a filepath to mapfile and returns the mean image
    as a numpy array.
    '''
    assert (isinstance(df, pd.DataFrame) or os.path.isfile(mapfile)), \
        'A valid mapfile or Pandas Dataframe are required!'

    # Set savepath
    if savepath is None:
        savepath = os.path.join(os.path.dirname(mapfile_utils.__file__),
                                                      '../meanfiles/')
        savepath = os.path.normpath(savepath)

    # Read mapfile
    if df is None:
        df, _, filename = mapfile_utils.read(mapfile,
                                             frac_sample=None)
        filename = os.path.splitext(filename)[0] + '.png'

    # Debug switch
    if debug_mode:
        df = df.ix[0:100]

    # TODO -- compute mean image for arbitrary array

    # Read first image and allocate empty array
    img = cv2.imread(df['filepath'].ix[0])
    mean_img = np.zeros(img.shape, dtype=np.float32)
    im_width, im_height, n_ch = mean_img.shape

    # Read images in dataframe
    for file_count, elem in enumerate(df.values):
        img = cv2.imread(elem[0])

        # Error check
        if img is None:
            raise ValueError('Error loading image {0:s}'.format(elem[0]))
        else:
            if file_count % 1000 == 0:
                print('Processed {0:d} images.'.format(file_count+1))

        # Resize image if necessary (default bilinear interpolation)
        if img.shape != mean_img.shape:
            img = cv2.resize(img, (im_width, im_height))

        # Accumulate image
        mean_img = cv2.accumulate(img, mean_img)

    # Update file_count to account for zero indexing
    file_count += 1
    print('Processed {0:d} images.\n'.format(file_count))

    # Optional, account for data augmentation such as rotation/ flip
    if data_aug:
        mean_img = mean_img + np.fliplr(mean_img) + \
                   np.rot90(mean_img, 1) + \
                   np.rot90(mean_img, 2) + \
                   np.rot90(mean_img, 3)
        divisor = file_count*5
    else:
        divisor = file_count

    # Divide accumulated array by total number of images
    mean_img = np.divide(mean_img, divisor)

    if save_img:
        # Make savepath if directory does not already exist
        if not os.path.isdir(savepath):
            os.makedirs(savepath)

        full_filepath = os.path.normpath(os.path.join(savepath, filename))
        print('Saving mean image to:\n\t{0:s}'.format(full_filepath))

        # Save mean image as png
        cv2.imwrite(full_filepath, mean_img)

        # Save mean image
        save_xml(full_filepath.replace('.png','_mean.xml'), mean_img,
                 img_width=im_width,
                 img_height=im_height,
                 n_ch=n_ch)

    return mean_img
            

    # Save intensity eigval, eigvec
#    save_xml(full_filepath.replace('.png','_intensity.xml'), mean_img,
#              img_width=im_width,
#              img_height=im_height,
#              n_ch=n_ch)

##
def save_xml(filename, mean_img, img_height=None, img_width=None, n_ch=None):
    '''
    compute_mean.save_xml(filename, img_height, img_width, n_ch)

    Saves a
    '''
#    assert (img_height and img_width and n_ch), \
#        raise RuntimeError('Input image height and width are required!')

    # Check filename extension
    filepath, ext = os.path.splitext(filename)
    if ext.lower() != '.xml':
        filename.replace(ext,'xml')

    filename = os.path.normpath(filename)
    print('Saving mean image xml to:\n\t{0}'.format(filename))

    n_pixels = img_height * img_width * n_ch

    root = etree.Element('opencv_storage')
    etree.SubElement(root, 'Channel').text = str(n_ch)
    etree.SubElement(root, 'Row').text = str(img_height)
    etree.SubElement(root, 'Col').text = str(img_width)
    meanImg = etree.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    etree.SubElement(meanImg, 'rows').text = '1'
    etree.SubElement(meanImg, 'cols').text = str(n_pixels)
    etree.SubElement(meanImg, 'dt').text = 'f'

    mean_img_string = []
    for idx, n in enumerate(np.reshape(mean_img, (n_pixels))):
        mean_img_string.append('{:e}'.format(n))
#        if (idx % 5)==0 and idx > 4:
#            mean_img_string.append('&apos&#xD;&#xA;&apos') # insert CRLF every 5 elems
                                                   # &#13;&#10;
                                                   # &#xD;&#xA;

    mean_img_string = ' '.join(mean_img_string) # white space:
                                               #     \u0020
                                               #     &#160;

    etree.SubElement(meanImg, 'data').text = mean_img_string

#    cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    f = open(filename, 'w')
    f.writelines(etree.tostring(root, encoding='UTF-8',
                                xml_declaration=True,
                                method='xml',
                                pretty_print=True).decode('UTF-8'))
    f.close()

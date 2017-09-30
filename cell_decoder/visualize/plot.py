# cell_decoder/visualize/plot.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ==============================================================================
'''
Cell DECODER plotting functions.

Copyright (c) 2017 Jeffrey J. Nirschl
'''

# Imports
import os
import re
import time
import numpy as np
import pandas as pd

# Third-party imports
import bokeh
import cv2
import numpy as np
import random as rnd
import holoviews as hv

# Plotting inports
import plotly.plotly as py
import plotly.graph_objs as go

# Import cell_decoder
from cell_decoder.io import mapfile_utils
from cell_decoder.img import transforms
from cell_decoder.visualize import colormaps

# Specify opencv optimization
cv2.setUseOptimized(True)

## Plot the range of possible data augmentation
def data_augmentation(mapfile):
    '''data_augmentation

    Accepts X and returns y.
    '''
    return


##
def model_weights(model_path, mapfile):
    '''visualize

    Accepts X and returns y.
    '''

    return


# Plot images using holoviews
def unique_images(mapfile,
                  max_size=224,
                  randomize=True,
                  save_im = True,
                  text_labels=None,
                  unique=True):
    '''unique_images

    Accepts X and returns Y.
    '''
    # Read mapfile
    _, df_sample = mapfile_utils.sample(mapfile,
                                        n_sample=1,
                                        replace=False)

    # Check whether images exist
    mapfile_utils.check_images(df=df_sample,
                               n_sample='all')

    # Pre-allocate output list
    hv_img = []
    for idx, (filepath, label) in enumerate(df_sample.values):
        if text_labels:
            tmp_label = text_labels[idx]
        else:
            tmp_label = str(label)

        bgr_img = cv2.imread(filepath)

        #TODO setup resize vs crop image
        if np.any(np.greater_equal(bgr_img.shape[0:2], max_size)):
            temp_img, out_dims = transforms.resize(bgr_img,width=max_size)
#            print('Resizing images from {0:d}x{1:d} to {2:d}x{3:d}'.format(zip(out_dims, [max_size, max_size])))
            temp_img = cv2.resize(bgr_img, (max_size, max_size),
                                  interpolation=cv2.INTER_AREA) # Use cv2.INTER_AREA for shrinking
#            print('Resizing images from {0:d}x{1:d} to {2:d}x{3:d}'.format(out_dims[0],out_dims[1], max_size, max_size)))
            rgb_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # Append to list of holoviews images
        if len(bgr_img.shape)==3:
            hv_img.append(hv.RGB(rgb_img, label=tmp_label))
        else:
            hv_img.append(hv.Image(rgb_img, label=tmp_label))

    return hv_img, df_sample

##
def prepare_plotly(df,
                   title=None,
                   size=5,
                   cmap=None,
                   cscale=None,
                   opacity=0.9,
                   mode='markers',
                   hovermode='closest',
                   text_labels=None):
    '''
    prepare_plotly
    
    
    '''
    if cmap is None:
        num_classes = max(df['label'].unique())
        if num_classes <= 10:
            cmap = colormaps.rgb('sasha_20')
#            cmap = bokeh.palettes.Category10
        elif num_classes <= 20:
            cmap = colormaps.rgb('sasha_20')
#            cmap = bokeh.palettes.Category20
        else:
            cmap = colormaps.rgb('sasha_20')

    # Prepare a Plotly trace for each group
    trace_list = []

    df_grouped = df.groupby('label')

    count = 0
    for k, sub_df in df_grouped:
        temp_trace = go.Scatter3d(x=sub_df['tsne_1'],
                                  y=sub_df['tsne_2'],
                                  z=sub_df['tsne_3'],
                                  mode=mode,
                                  marker=dict(
                                      size=size,
                                      color=cmap[count][1], #df['label']
                                      #colorscale=cmap,   # choose a colorscale
                                      opacity=opacity ),
                                  name=text_labels,
                                  text=sub_df['label']
        )

        # Append to list
        trace_list.append(temp_trace)
        count +=1

    # Create layout
    layout= go.Layout(
        title=title,
        hovermode=hovermode,
        xaxis= dict(
            title='test',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title= 'Rank',
            ticklen= 5,
            gridwidth= 2,
        ),
        showlegend=True
    )

    fig= go.Figure(data=trace_list, layout=layout)
    
    return fig

##
def prepare_holoviews(df):
    '''
    
    '''

    Layout = 1
    return Layout

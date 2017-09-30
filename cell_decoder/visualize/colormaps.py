# cell_decoder/visualize/colormaps.py
#
# Copyright (c) 2017 Jeffrey J. Nirschl
# All rights reserved
# All contributions by Jeffrey J. Nirschl in the
# laboratory of Erika Holzbaur at the University of Pennsylvania.
#
# Distributable under an MIT License
# ==============================================================================
'''
Cell DECODER colormap functions.

Copyright (c) 2017 Jeffrey J. Nirschl
'''

# Imports
import os
import re
import time
import numpy as np
import pandas as pd

# Third-party imports
import cv2
import numpy as np
import random as rnd
import holoviews as hv

# Import cell_decoder
from cell_decoder.io import mapfile_utils
from cell_decoder.img import transforms

# Specify opencv optimization
cv2.setUseOptimized(True)

##
#def distinguishable_colors()

# Copied from
# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
# Colormap from sasha t for plotly
# colorscale = cmap
def rgb(name):
    '''
    rgb
    
    '''
    defaults = {'sasha_20':[[0, 'rgb(230, 25, 75)'],
                            [0.04762, 'rgb(60, 180, 75)'],
                            [0.09524, 'rgb(255, 225, 25)'],
                            [0.14286, 'rgb(0, 130, 200)'],
                            [0.19048, 'rgb(245, 130, 48)'],
                            [0.2381, 'rgb(145, 30, 180)'],
                            [0.28571, 'rgb(70, 240, 240)'],
                            [0.33333, 'rgb(240, 50, 230)'],
                            [0.38095, 'rgb(210, 245, 60)'],
                            [0.42857, 'rgb(250, 190, 190)'],
                            [0.47619, 'rgb(0, 128, 128)'],
                            [0.52381, 'rgb(230, 190, 255)'],
                            [0.57143, 'rgb(170, 110, 40)'],
                            [0.61905, 'rgb(255, 250, 200)'],
                            [0.66667, 'rgb(128, 0, 0)'],
                            [0.71429, 'rgb(170, 255, 195)'],
                            [0.7619, 'rgb(128, 128, 0)'],
                            [0.80952, 'rgb(255, 215, 180)'],
                            [0.85714, 'rgb(0, 0, 128)'],
                            [0.90476, 'rgb(128, 128, 128)'],
                            [0.95238, 'rgb(255, 255, 255)'],
                            [1, 'rgb(224, 224, 224)']]
    }
    
    return defaults[name]

## 
def hex():
    '''
    colormaps.hex
    
    '''
    defaults = {'sasha_20':[[0, 'rgb(230, 25, 75)'],
                            [0.04762, 'rgb(60, 180, 75)'],
                            [0.09524, 'rgb(255, 225, 25)'],
                            [0.14286, 'rgb(0, 130, 200)'],
                            [0.19048, 'rgb(245, 130, 48)'],
                            [0.2381, 'rgb(145, 30, 180)'],
                            [0.28571, 'rgb(70, 240, 240)'],
                            [0.33333, 'rgb(240, 50, 230)'],
                            [0.38095, 'rgb(210, 245, 60)'],
                            [0.42857, 'rgb(250, 190, 190)'],
                            [0.47619, 'rgb(0, 128, 128)'],
                            [0.52381, 'rgb(230, 190, 255)'],
                            [0.57143, 'rgb(170, 110, 40)'],
                            [0.61905, 'rgb(255, 250, 200)'],
                            [0.66667, 'rgb(128, 0, 0)'],
                            [0.71429, 'rgb(170, 255, 195)'],
                            [0.7619, 'rgb(128, 128, 0)'],
                            [0.80952, 'rgb(255, 215, 180)'],
                            [0.85714, 'rgb(0, 0, 128)'],
                            [0.90476, 'rgb(128, 128, 128)'],
                            [0.95238, 'rgb(255, 255, 255)'],
                            [1, 'rgb(0, 0, 0)']]
    }


    
#Hex	RGB	CMYK
#Red	#e6194b	(230, 25, 75)	(0, 100, 66, 0)
#Green	#3cb44b	(60, 180, 75)	(75, 0, 100, 0)
#Yellow	#ffe119	(255, 225, 25)	(0, 25, 95, 0)
#Blue	#0082c8	(0, 130, 200)	(100, 35, 0, 0)
#Orange	#f58231	(245, 130, 48)	(0, 60, 92, 0)
#Purple	#911eb4	(145, 30, 180)	(35, 70, 0, 0)
#Cyan	#46f0f0	(70, 240, 240)	(70, 0, 0, 0)
#Magenta	#f032e6	(240, 50, 230)	(0, 100, 0, 0)
#Lime	#d2f53c	(210, 245, 60)	(35, 0, 100, 0)
#Pink	#fabebe	(250, 190, 190)	(0, 30, 15, 0)
#Teal	#008080	(0, 128, 128)	(100, 0, 0, 50)
#Lavender	#e6beff	(230, 190, 255)	(10, 25, 0, 0)
#Brown	#aa6e28	(170, 110, 40)	(0, 35, 75, 33)
#Beige	#fffac8	(255, 250, 200)	(5, 10, 30, 0)
#Maroon	#800000	(128, 0, 0)	(0, 100, 100, 50)
#Mint	#aaffc3	(170, 255, 195)	(33, 0, 23, 0)
#Olive	#808000	(128, 128, 0)	(0, 0, 100, 50)
#Coral	#ffd8b1	(255, 215, 180)	(0, 15, 30, 0)
#Navy	#000080	(0, 0, 128)	(100, 100, 0, 50)
#Grey	#808080	(128, 128, 128)	(0, 0, 0, 50)
#White	#FFFFFF	(255, 255, 255)	(0, 0, 0, 0)
#Black	#000000	(0, 0, 0)	(0, 0, 0, 100




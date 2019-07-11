#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:41:21 2019

@author: marcelo
"""
import numpy as np
import os
from skimage.feature import hog
from skimage import io
from skimage import data, img_as_float
from skimage import exposure

PARAM = {
        'orientations': np.array(np.arange(4, 15)),
        'pixels_per_cell': np.array([[10, 10], [20, 20], [30, 30], [40, 40]]) ,
        'cells_per_block': np.array([[1, 1], [2, 2], [3, 3], [4, 4]]),
        }

if __name__ == '__main__':
    path = '/home/marcelo/Documentos/inteligencia_computacional_disciplina/Tissue_Class/01_TUMOR/1A11_CRC-Prim-HE-07_022.tif_Row_601_Col_151.tif'
    img = io.imread(path, as_gray=True)
    img_test = data.moon()
    print(type(img_test))
    
    img_adapteq = exposure.equalize_adapthist(img_test, clip_limit=0.03)

    print(img.shape)
    
    io.imshow(img_adapteq)
    io.show()
       
    for orient in PARAM['orientations']:
        for pixel_cell in PARAM['pixels_per_cell']:
            for cell_block in PARAM['cell_block']:
                break
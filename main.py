#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:38:14 2019

@author: marcelo
"""

import numpy as np
import os
import cv2
import pandas as pd
import random
import string
import tqdm
from skimage.feature import hog

PARAM = {
        'orientations': np.arange(6, 11, 2),
        'pixels_per_cell': np.array([[5,5], [10,10]]),
        'cells_per_block': np.array([[1, 1], [2, 2]])
        }

def get_images(path):
    return (cv2.imread(os.path.join(path, name_img), cv2.IMREAD_GRAYSCALE) for name_img in tqdm.tqdm(os.listdir(path)))

def save_file(arr_hog, path_hog):
    
    df_hog = pd.DataFrame(np.array(arr_hog))
    
    df_hog.to_csv(path_hog, index=False)

def get_name_random():
    aux_name = ''
    aux_rand_name = random.sample(string.ascii_uppercase, 8)
    aux_name = aux_name.join(aux_rand_name)
    aux_name += '.csv'
    return aux_name

def create_data_hog(DATADIR, OUTDIR, orientation, pixel_cell, cell_block):
         
    CATEGORIES = os.listdir(DATADIR)

    for category in CATEGORIES:
        
        print('Iniciando trabalho na Categoria: ', category)
        
        path = os.path.join(DATADIR, category)
        imgs = get_images(path)
        
        eq_adap_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
        
        for img in tqdm.tqdm(imgs):

            path_hog = OUTDIR + '/' + category + '/' + 'orient_' + str(orientation) + '_ppc_' + str(pixel_cell[0]) + '_cpb_' + str(cell_block[0]) + '/'
        
            if not os.path.exists(path_hog):
                os.makedirs(path_hog)
        
            
            path_hog += get_name_random()
            
            img_equ_adap = eq_adap_obj.apply(img)
            
            hog_feature = hog(img_equ_adap, orientations=orientation, pixels_per_cell=pixel_cell, cells_per_block=cell_block, block_norm="L1")
        
            save_file(hog_feature, path_hog)
    
def main():

    DATADIR = "/home/marcelo/Documentos/inteligencia_computacional_disciplina/Tissue_Class"

    OUTDIR = "hog_features"

    for orient in PARAM['orientations']:
            for ppc in PARAM['pixels_per_cell']:
                for cpb in PARAM['cells_per_block']:
                    
                    if ppc[0]==5 and cpb[0] == 2:
                        continue
                    
                    print('_______________________________________________')
                    print('orientations: ', orient)
                    print('pixels_per_cell: ', ppc)
                    print('cells_per_block: ', cpb)
                    create_data_hog(DATADIR, OUTDIR, orientation = orient, \
                                    pixel_cell = ppc, cell_block = cpb)
    
    print('Finalizado...')
    
if __name__ == '__main__':
    
    main()

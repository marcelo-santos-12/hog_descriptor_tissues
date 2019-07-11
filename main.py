import numpy as np
import os
#import cv2
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog

import params

print(params.PARAM)

def dir_image(dir_):
    '''
    Pega o nome das subpastas das imagens histologicas
    param dir_: caminho completo do diretorio que contem as pastas das imagens
    return: uma lista contendo o nome das subpastas  
    '''
    categories = os.listdir(dir_)
    print(categories)
    return categories

def create_training_data():
    descriptors = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array                
                features = hog(img_array, orientations=4, pixels_per_cell=(50, 50), cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1")
                descriptors.append(features)
            except Exception as e:
                print(e)

    return descriptors

def entry_dir(dir_tissue):
    pass

def main():
    path = '/home/marcelo/Documentos/inteligencia_computacional_disciplina/Tissue_Class'
    subfolders = dir_image(path)
    
    for cat in subfolders:
        dir_imgs = os.path.join(path, cat)
        name_imgs = os.listdir(dir_imgs)
        print(name_imgs)
        
        
    

    

if __name__ == '__main__':
    
    main()
    
    
import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog

PARAM = {
        'orientations': np.array(np.arange(6, 11, 2)),
        'pixels_per_cell': np.array([[5,5], [10,10]]) ,
        'cells_per_block': np.array([[1, 1], [2, 2]])
        }

def create_data_hog(DATADIR, OUTDIR, orientation, pixel_cell, cell_block):
    
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    
    CATEGORIES = os.listdir(DATADIR)
    data = []
    class_ = []
    
    eq_adap_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # get the classification
        for name_img in tqdm(os.listdir(path)):
            try:
                img = cv2.imread(os.path.join(path, name_img), cv2.IMREAD_GRAYSCALE)  # convert to array
    
                img_equ_adap = eq_adap_obj.apply(img)
    
                feature = hog(img_equ_adap, orientations=orientation, pixels_per_cell=pixel_cell, cells_per_block=cell_block, block_norm="L1")

                data.append(feature)
                class_.append(class_num)

            except Exception as e:
                print(e)

    #SALVANDO DADOS
    columns = ['COL' + str(indice) for indice in np.arange(len(data[0]))]

    df_train = pd.DataFrame(np.array(data), columns=columns)
    
    df_class = pd.DataFrame(np.array(class_), columns=['class'])

    df_data = df_train.join(df_class)

    path_csv = OUTDIR + '/' + 'hog_' + 'orient_' + str(orientation) + '_' + 'p_p_c_' + str(pixel_cell[0]) + '_' + 'c_p_b_' + str(cell_block[0]) +'.csv'
    
    df_data.to_csv(path_csv, index=False)

def main():

    DATADIR = "/home/marcelo/Documentos/inteligencia_computacional_disciplina/Tissue_Class"

    OUTDIR = "hog_features"

    for orient in PARAM['orientations']:
            for ppc in PARAM['pixels_per_cell']:
                for cpb in PARAM['cells_per_block']:
                    
                    print('orientations: ', orient)
                    print('pixels_per_cell: ', ppc)
                    print('cells_per_block: ', cpb)
                    create_data_hog(DATADIR, OUTDIR, orientation = orient, \
                                    pixel_cell = ppc, cell_block = cpb)
                    
                    print('_________________________________')
    
    print('Finalizado...')
    
if __name__ == '__main__':
    
    main()
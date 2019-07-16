import cv2
from skimage.feature import hog
import params
import glob
import numpy as np

PARAM = {
        'orientations': np.array(np.arange(6, 11, 2)),
        'pixels_per_cell': np.array([[5,5], [10,10]]) ,
        'cells_per_block': np.array([[1, 1], [2, 2]])
        }

def show_img_hog(img, orientation, pixel_cell, cell_block):
    _, img_hog = hog(img, visualize=True, orientations=orientation, pixels_per_cell=pixel_cell, cells_per_block=cell_block, transform_sqrt=True, block_norm="L1")
    name_window = 'Orientation: ' + str(orientation) + ' | Pixel per cell ' + str(pixel_cell) + ' | Cells per block: ' + str(cell_block)
    
    img_show = np.hstack([img, img_hog])
    cv2.imshow(name_window, img_show)

    if cv2.waitKey(0) == ord('w'):

        cv2.destroyAllWindows()
        quit()
    
    else:
        
        cv2.destroyAllWindows()
    
def main():
    
    path = 'img_sample'
    
    for name_img in glob.iglob(path + '/*.tif'):
        img = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
        cont_hog = 0
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        for orient in params.PARAM['orientations']:
            for ppc in params.PARAM['pixels_per_cell']:
                for cpb in params.PARAM['cells_per_block']:
                    cont_hog += 1
                    print('orientations: ', orient)
                    print('pixels_per_cell: ', ppc)
                    print('cells_per_block: ', cpb)
                    show_img_hog(img, orient, ppc, cpb)
                    print('_________________________________')
        print('QTD de HOGs: ', cont_hog)
            
        break

def visualize_equ_hist():
    
    img = cv2.imread('img_sample/1DAA_CRC-Prim-HE-05_026.tif_Row_151_Col_1.tif', 0)
    
    eq_adap_obj = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
    img_equ_adap = eq_adap_obj.apply(img)
    img_equ_loc = cv2.equalizeHist(img)
    
    img_join = np.hstack([img, img_equ_adap, img_equ_loc])
    
    cv2.imshow('test', img_join)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    quit()

if __name__ == '__main__':    
    
    if True:
        visualize_equ_hist()
    
    main()

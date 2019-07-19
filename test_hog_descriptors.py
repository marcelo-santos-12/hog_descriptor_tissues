import cv2
import glob
import numpy as np

from skimage.feature import hog
from skimage import exposure

PARAM = {
        'orientations': np.arange(6, 11, 2),
        'pixels_per_cell': np.array([[5,5], [10,10]]) ,
        'cells_per_block': np.array([[1, 1], [2, 2]])
        }

def show_img_hog(img, orientation, pixel_cell, cell_block):
    features, img_hog = hog(img, visualize=True, orientations=orientation, pixels_per_cell=pixel_cell, cells_per_block=cell_block, transform_sqrt=True, block_norm="L1")
    name_window = 'Orientation: ' + str(orientation) + ' | Pixel per cell ' + str(pixel_cell) + ' | Cells per block: ' + str(cell_block)
    
    print('Comprimento: ', len(features))
    
    img_hog = cv2.resize(img_hog, (450, 450))
    
    img = cv2.resize(img, (450, 450))
    
    img_hog_rescaled = exposure.rescale_intensity(img_hog, out_range=(0, 255))
    img_hog_rescaled = img_hog_rescaled.astype("uint8")
    
    cv2.imwrite('img_article/orient_' + str(orientation) + '_ppc ' + str(pixel_cell[0]) + '_cpb_' + str(cell_block[0]) + '.jpg', img_hog_rescaled)
    
    img_show = np.hstack([img, img_hog_rescaled])

    cv2.imshow(name_window, img_show)

    if cv2.waitKey(0) == ord('w'):

        cv2.destroyAllWindows()
        quit()
    
    else:
        
        cv2.destroyAllWindows()
    
def main(path):
    
    
    for name_img in glob.iglob(path + '/*.tif'):
        print('Imagem atual: ', name_img[11:])
        img = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
        
        visualize_equ_hist(name_img[11:], img)
        
        for orient in PARAM['orientations']:
            for ppc in PARAM['pixels_per_cell']:
                for cpb in PARAM['cells_per_block']:
                    
                    if ppc[0] == 5 and cpb[0] == 2:
                            continue
                    
                    
                    print('orientations: ', orient)
                    print('pixels_per_cell: ', ppc)
                    print('cells_per_block: ', cpb)

                    show_img_hog(img, orient, ppc, cpb)
        
                    print('_________________________________')

        break
    
def visualize_equ_hist(name_img, img):
    obj_eq_adap = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
    img_equ_adap = obj_eq_adap.apply(img)
    
    img = cv2.resize(img, (450, 450))
    img_equ_adap = cv2.resize(img_equ_adap, (450, 450))
    
    print('img_article/cinza_' + name_img[:-4] + '.png')
    cv2.imwrite('img_article/cinza_' + name_img[:-4] + '.png', img)
    cv2.imwrite('img_article/equ_' + name_img[:-4] + '.png', img_equ_adap)
    
    img_join = np.hstack([img, img_equ_adap])
    
    cv2.imshow(name_img, img_join)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':    
     
    path = 'img_sample' #diretorio das imagens de amostra
    main(path)

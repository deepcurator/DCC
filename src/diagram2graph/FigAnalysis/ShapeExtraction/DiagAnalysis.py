from __future__ import print_function
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import glob
import os

from ShapeDetect import ShapeDetect as sd
from ArrowDetect import ArrowDetect as ad
from TextDetect_OPENCV import TextDetect_EAST as tda
from Text2Graph import Text2Graph as tg

def preprocessImage(image_path, resize):
    
        # load the image on disk and then display it
        image = cv2.imread(image_path)
        image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255, 255, 255]) # add white border in the original image
        
        if resize == 1:
            newX, newY = image.shape[1]*1.5, image.shape[0]*1.5
        else:
            newX, newY = image.shape[1], image.shape[0]
            
        image_resize = cv2.resize(image,(int(newX),int(newY)))
        
#        cv2.imshow("Original", image)
#        cv2.waitKey(0)
        
        imgPIL = Image.open(image_path)
        imgPIL = ImageOps.expand(imgPIL, border = 10, fill = 'white')
        imgPIL = imgPIL.resize((int(newX),int(newY)), Image.ANTIALIAS)
        
        imgPIL = ImageEnhance.Color(imgPIL)
        imgPIL = imgPIL.enhance(0)
        gray_im = imgPIL.convert('L')  # convert image to monochrome
        # gray_im = ImageEnhance.Sharpness(gray_im).enhance(2)
        # gray_im = ImageEnhance.Contrast(gray_im).enhance(2)
    
        gray_imcv = np.array(gray_im, dtype=np.uint8)
    
        _, thresh_im = cv2.threshold(gray_imcv, 240, 255, cv2.THRESH_BINARY_INV)
    
        #cv2.imshow("gray_im", np.array(gray_im, dtype=np.uint8))
        #cv2.waitKey(0)
    
        #cv2.imshow("thresh_im", thresh_im)
        #cv2.waitKey(0)
        # Find edges in the image using canny edge detection method
        # Calculate lower threshold and upper threshold using sigma = 0.33
        # sigma = 0.33
        # v = np.median(thresh_im)
        # low = int(max(0, (1.0 - sigma) * v))
        # high = int(min(255, (1.0 + sigma) * v))
    
        # edged = cv2.Canny(thresh_im, low, high)
        
        return image_resize, thresh_im, gray_imcv
        
        
if __name__ == '__main__':
    imdir_path = '/mnt/vtsraid01/Aditi/home/DiagramAnalysis/DATASET_MULTICLASS_v1.0/Validation/2D'
    op_dir = "/home/z003yk0t/DiagramAnalysis/ContentExtraction/ShapeExtraction/2Dresults_validation"
        
    print("[INFO] loading images ...")
    
    for filename in glob.glob(os.path.join(imdir_path, '*png')):
        print(filename)
        im, thresh_im, gray_imcv = preprocessImage(filename, 0)
        
        
        shapedetector = sd()
        component = shapedetector.find_component(filename, op_dir, im, thresh_im, gray_imcv)
    #    for c in component:
    #        cv2.drawContours(image, [c], 0, (0, 255, 0), 2)
    #        
                
        textdetector = tda()
        text_list = textdetector.combinedTextDetect(filename, im)
#    
#        arrowdetector = ad()            
#        merged_lines = arrowdetector.detectLines(im, thresh_im, gray_imcv, component, text_list)
    
        graphcreator = tg()
        graph = graphcreator.createText2Graph(im, component, text_list)
    #    cv2.imshow("image", image)
    #    cv2.waitKey(0) 
    
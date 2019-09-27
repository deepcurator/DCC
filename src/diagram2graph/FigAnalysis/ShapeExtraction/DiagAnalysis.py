from __future__ import print_function
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import glob
import os

from ShapeDetect import ShapeDetect as sd
from ArrowDetect import ArrowDetect as ad
from TextDetect_OPENCV import TextDetectAll as tda
#from Diag2Graph import Diag2Graph as tg
from Diag2Graph_v2 import Diag2Graph as tgv2
from ParseJSON import ParseJSON as pj


def preprocessImage(image_path, resize):
    
        # load the image from disk and then preprocess it
        image = cv2.imread(image_path)
        image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255, 255, 255]) # add white border in the original image
        
        if resize == 1:
            newX, newY = image.shape[1]*1.5, image.shape[0]*1.5
        else:
            newX, newY = image.shape[1], image.shape[0]
            
        image_resize = cv2.resize(image,(int(newX),int(newY)))
        
        imgPIL = Image.open(image_path)
        imgPIL = ImageOps.expand(imgPIL, border = 10, fill = 'white')
        imgPIL = imgPIL.resize((int(newX),int(newY)), Image.ANTIALIAS)        
        imgPIL = ImageEnhance.Color(imgPIL)
        imgPIL = imgPIL.enhance(0)
        gray_im = imgPIL.convert('L') 

        gray_imcv = np.array(gray_im, dtype=np.uint8)    
        _, thresh_im = cv2.threshold(gray_imcv, 240, 255, cv2.THRESH_BINARY_INV)    
        
        return image_resize, thresh_im, gray_imcv
        
        
if __name__ == '__main__':
    
    imdir_path = 'InputImage'
    op_dir = "Output"
        
    print("[INFO] loading images ...")
    
    for filename in glob.glob(os.path.join(imdir_path, '*png')):
        print(filename)
        im, thresh_im, gray_imcv = preprocessImage(filename, 0)
        
        parsejson = pj()
        paper_title, paper_file_name, paper_conf, paper_year, fig_caption, fig_text = parsejson.getCaption(filename)
        if not os.path.isdir(os.path.join(op_dir, paper_file_name)):
            os.mkdir(os.path.join(op_dir, paper_file_name))
            os.mkdir(os.path.join(op_dir, paper_file_name, "diag2graph"))
            os.mkdir(os.path.join(op_dir, paper_file_name, "Figures"))

        cv2.imwrite(os.path.join(op_dir, paper_file_name+ "/Figures/" + os.path.basename(filename)), im)        
        
        shapedetector = sd()
        component, flow_dir = shapedetector.find_component(filename, op_dir, im, thresh_im, gray_imcv)
               
        textdetector = tda()
        text_list = textdetector.combinedTextDetect(filename, im, component, fig_text)
        
        arrowdetector = ad()            
        line_connect = arrowdetector.detectLines(im, thresh_im, gray_imcv, component, text_list)
    
        graphcreator = tgv2()
        graphcreator.createDiag2Graph(op_dir, filename, im, thresh_im, component, flow_dir, text_list, line_connect, paper_title, paper_file_name, paper_conf, paper_year, fig_caption)
        
        

import cv2
import pytesseract
import imutils
from PIL import Image, ImageEnhance, ImageFilter
import re
import numpy as np
import glob
import os
import argparse


def preprocessImage(image_file):
    
    imgPIL = Image.open(image_file)
    imgCV = cv2.imread(image_file)
    imgCV = cv2.resize(imgCV, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_CUBIC)
#    
#    imgCV = cv2.fastNlMeansDenoisingColored(imgCV,None,3,3,3,15)
#    grayIm = cv2.cvtColor(imgCV, cv2.COLOR_BGR2GRAY)
#    #grayIm = cv2.bilateralFilter(grayIm, 5, 20, 20)
#    ##grayIm = cv2.medianBlur(grayIm, 3)
#    gaussian_im = cv2.GaussianBlur(grayIm, (3,3), 10.0)
#    grayIm = cv2.addWeighted(grayIm, 1.5, gaussian_im, -0.5, 0, grayIm)

    width, height = imgPIL.size
    
    imgPIL = ImageEnhance.Color(imgPIL)
    imgPIL = imgPIL.enhance(0)
    #conver to gray scale
    grayIm = imgPIL.convert('L')
    #resize image
    grayIm = grayIm.resize((int(width*1.5),int(height*1.5)), Image.ANTIALIAS)
    #grayIm.show()
    grayIm = ImageEnhance.Brightness(grayIm).enhance(0.75)
    grayIm = ImageEnhance.Contrast(grayIm).enhance(2)
    grayIm = ImageEnhance.Sharpness(grayIm).enhance(2)
    #grayIm.show()
    #grayIm = grayIm.filter(ImageFilter.UnsharpMask(radius=2, percent=3, threshold=0))
    grayIm = np.array(grayIm) 
    #grayIm = cv2.adaptiveThreshold(grayIm,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
   
#    kernel = np.ones((2, 2), np.uint8)
#    grayIm = cv2.erode(grayIm, kernel, iterations=2)
#    grayIm = cv2.dilate(grayIm, kernel, iterations=2)
#        
#    cv2.imshow("grayImafter", grayIm)
#    cv2.waitKey(0)
    return imgCV, grayIm


def read_text(imgCV, grayIm):
    
    output_dict = pytesseract.image_to_data(grayIm, lang='eng', config='--psm 6 --oem 1', output_type='dict')
    
    text_list = output_dict.get('text')
    box_pos_left_corner = output_dict.get('left')
    box_pos_top_corner = output_dict.get('top')
    #box_pos_width = output_dict.get('width')
    #box_pos_height = output_dict.get('height')
    height, width, channels = imgCV.shape
        
    for index in range(len(text_list)):
        #print(text_list[index])
        op = re.sub(r'^[^0-9a-zA-Z]*', '', re.sub(r'[^0-9a-zA-Z]*$', '', text_list[index]))
        if not op:
            print("")
        else:
            #print(op)
            
            if box_pos_left_corner[index] > 0 and box_pos_left_corner[index] < width:
                posx = box_pos_left_corner[index]
            elif box_pos_left_corner[index] <= 0:
                posx = 30
            else:
                posx = width-30
            if box_pos_top_corner[index] > 0 and box_pos_top_corner[index] < height:
                posy = box_pos_top_corner[index]
            elif box_pos_top_corner[index] <= 0:
                posy = 30
            else:
                posy = height-30
                
            cv2.putText(imgCV, op, (posx,posy), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)
            #cv2.rectangle(imgCV,(box_pos_left_corner[index],box_pos_top_corner[index]),
    		#    (box_pos_left_corner[index]+box_pos_width[index],box_pos_top_corner[index]+box_pos_height[index]),
    		#    (255, 255, 255),-1)
    
    #cv2.imshow("op image", imgCV)
    #cv2.waitKey(0)
    return imgCV

if __name__ == "__main__":
    
    # Using Argument Parser to get the location of image
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--inputdir', required=True, help='Path to input image dir')
    ap.add_argument("-o", "--outputdir", required=True, help='Path to output image dir')
 
    args = vars(ap.parse_args())
    indir_path = args["inputdir"] 
    outdir_path = args["outputdir"]
    
    for filename in glob.glob(os.path.join(indir_path, '*png')):
        print(filename)
        
        imgCV, grayIm = preprocessImage(filename)
        imgCV = read_text(imgCV, grayIm)
        imgCV = cv2.resize(imgCV, None, fx = 0.75, fy = 0.75, interpolation = cv2.INTER_AREA)    
        
        op_file_name = os.path.join(outdir_path, "op" + os.path.basename(filename))
        print(op_file_name)
        cv2.imwrite(op_file_name, imgCV)
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:53:30 2019

@author: z003yk0t
"""

import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import re
import os

class TextDetect:
    
    def __init__(self):
        self.rx = 1.5
        self.ry = 1.5
        self.size_th = 0.01
        self.min_conf_thresh = 10
        
    def preprocessImage(self, image):
        
        imgCV = image.copy()
        
        imgPIL = Image.fromarray(imgCV)
        imgCV = cv2.resize(imgCV, None, fx = self.rx, fy = self.ry, interpolation = cv2.INTER_CUBIC)
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
        grayIm = grayIm.resize((int(width*self.rx),int(height*self.ry)), Image.ANTIALIAS)
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


    def read_text(self, fileName, imgCV, grayIm):
        
        output_dict = pytesseract.image_to_data(grayIm, lang='eng', config='--psm 6 --oem 1', output_type='dict')
        #print(output_dict)
        text_list = output_dict.get('text')
        prob_list = output_dict.get('conf')
        box_pos_left_corner = output_dict.get('left')
        box_pos_top_corner = output_dict.get('top')
        box_pos_width = output_dict.get('width')
        box_pos_height = output_dict.get('height')
        height, width, channels = imgCV.shape
        final_results = []   
        for index in range(len(text_list)):
            #print(text_list[index])
            op = re.sub(r'^[^()+*-<>#={}[\]/\0-9a-zA-Z]*', '', re.sub(r'[^()+*-<>#={}[\]/\0-9a-zA-Z]*$', '', text_list[index]))
            op = re.sub('\W+','', op)
            size = box_pos_width[index] * box_pos_height[index]
            #print("Text:%s, Box size: %d, Image size: %d, Thres image Size: %d" %(op, size, np.size(imgCV, 0)*np.size(imgCV, 1), np.size(imgCV, 0)*np.size(imgCV, 1)*self.size_th))
            if not op:
                print("")
            elif size < (np.size(imgCV, 0)*np.size(imgCV, 1)*self.size_th) and prob_list[index] > self.min_conf_thresh:
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
                
                
                startX = int(box_pos_left_corner[index]/self.rx)
                startY = int(box_pos_top_corner[index]/self.ry)
                endX = int((box_pos_left_corner[index]+box_pos_width[index])/self.rx)
                endY = int((box_pos_top_corner[index]+box_pos_height[index])/self.ry)
                prob = prob_list[index]
                
                
                final_results.append(((startX, startY, endX, endY), op, prob))
                cv2.putText(imgCV, op, (posx,posy), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)
                cv2.rectangle(imgCV,(box_pos_left_corner[index],box_pos_top_corner[index]),
        		    (box_pos_left_corner[index]+box_pos_width[index],box_pos_top_corner[index]+box_pos_height[index]),(0, 0, 255),1)
              
        imgCV = cv2.resize(imgCV, None, fx = 1/self.rx, fy = 1/self.ry, interpolation = cv2.INTER_AREA)   
#        cv2.imshow("Text Detect Whole Image", imgCV)
#        cv2.waitKey(0)
#        op_dir = "/home/z003yk0t/DiagramAnalysis/ContentExtraction/ShapeExtraction/2Dresults_validation/TextWholeIm"
#        op_file_name = os.path.join(op_dir, "op" + os.path.basename(fileName))
#        print(op_file_name)
#        cv2.imwrite(op_file_name, imgCV)
        return final_results

    def getText(self, fileName, image):
        imgCV, grayIm = self.preprocessImage(image)
        output_dict = self.read_text(fileName, imgCV, grayIm)
        return output_dict
#        op_file_name = os.path.join(outdir_path, "op" + os.path.basename(filename))
#        print(op_file_name)
#        cv2.imwrite(op_file_name, imgCV)
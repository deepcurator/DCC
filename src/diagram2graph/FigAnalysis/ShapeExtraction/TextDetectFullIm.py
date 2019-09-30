import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import re
import os
from SpellCorrect import SpellCorrect as sc



class TextDetectFullIm:
    
    def __init__(self):
        self.rx = 1.5
        self.ry = 1.5
        self.size_th = 0.01
        self.min_conf_thresh = 10
        
    def preprocessImage(self, image):
        
        imgCV = image.copy()
        
        imgPIL = Image.fromarray(imgCV)
        imgCV = cv2.resize(imgCV, None, fx = self.rx, fy = self.ry, interpolation = cv2.INTER_CUBIC)
        width, height = imgPIL.size
        
        imgPIL = ImageEnhance.Color(imgPIL)
        imgPIL = imgPIL.enhance(0)
        grayIm = imgPIL.convert('L')
        grayIm = grayIm.resize((int(width*self.rx),int(height*self.ry)), Image.ANTIALIAS)
        grayIm = ImageEnhance.Brightness(grayIm).enhance(0.75)
        grayIm = ImageEnhance.Contrast(grayIm).enhance(2)
        grayIm = ImageEnhance.Sharpness(grayIm).enhance(2)
        grayIm = np.array(grayIm) 
        
        return imgCV, grayIm


    def read_text(self, fileName, imgCV, grayIm, cfig, fig_text):
        
        spellcorr = sc()
        output_dict = pytesseract.image_to_data(grayIm, lang='eng', config=cfig, output_type='dict')
        
        text_list = output_dict.get('text')
        prob_list = output_dict.get('conf')
        box_pos_left_corner = output_dict.get('left')
        box_pos_top_corner = output_dict.get('top')
        box_pos_width = output_dict.get('width')
        box_pos_height = output_dict.get('height')
        height, width, channels = imgCV.shape
        final_results = []   
        for index in range(len(text_list)):
            
            op = re.sub(r'^[^()+*-<>#=&{}[\]/\0-9a-zA-Z]*', '', re.sub(r'[^()+*-<>#=&{}[\]/\0-9a-zA-Z]*$', '', text_list[index]))
            op = re.sub('\W+','', op)
            size = box_pos_width[index] * box_pos_height[index]
#            if not op:
#                print("")
            if op and (size < (np.size(imgCV, 0)*np.size(imgCV, 1)*self.size_th) and prob_list[index] > self.min_conf_thresh):
                
                op_cor = spellcorr.correctWord(op, fig_text)
                #print("Full IM: op = %s, op_cor %s"%(op, op_cor))
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
                final_results.append(((startX, startY, endX, endY), op_cor, prob))
               
                cv2.putText(imgCV, op_cor, (posx,posy), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)
                cv2.rectangle(imgCV,(box_pos_left_corner[index],box_pos_top_corner[index]),
        		    (box_pos_left_corner[index]+box_pos_width[index],box_pos_top_corner[index]+box_pos_height[index]),(0, 0, 255),1)
              
        # imgCV = cv2.resize(imgCV, None, fx = 1/self.rx, fy = 1/self.ry, interpolation = cv2.INTER_AREA)   
        # cv2.imshow("Text Detect Whole Image", imgCV)
        # cv2.waitKey(0)
        return final_results

    def getText(self, fileName, image, cfig, fig_text):
        imgCV, grayIm = self.preprocessImage(image)
        output_dict = self.read_text(fileName, imgCV, grayIm, cfig, fig_text)
        return output_dict

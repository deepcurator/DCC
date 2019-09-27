from __future__ import print_function
import numpy as np
import cv2
import pytesseract
from imutils.object_detection import non_max_suppression    
from PIL import Image, ImageEnhance, ImageFilter
#from TextDetect import TextDetect as td
import re
from Rectangle import Rectangle
from RectangleMerger import RectangleMerger as Merger

import difflib
import os
from scipy import ndimage

from SpellCorrect import SpellCorrect as sc



class TextDetectEast:
    
    def __init__(self):
        # parameters for EAST detector
        self.min_confidence = 0.3
        self.size = 320
        self.padding = 0.0
        self.overlapTh = 0.65
        self.tess_config = ("-l eng --oem 1 --psm 11")
    
    
    def boxToRec(self, box):
        rec = []
        for x1, y1, x2, y2, in box:
            r = Rectangle.from_2_pos(x1, y1, x2, y2)
            rec.append(r)
        return rec  
                   
    def decodePredictions(self, scores, geometry):
        (rows, cols) = scores.shape[2:4]  # grab the rows and columns from score volume
        rects = []  # stores the bounding box coordiantes for text regions
        confidences = []  # stores the probability associated with each bounding box region in rects
        
        for y in range(rows):
            scoresdata = scores[0, 0, y]
            xdata0 = geometry[0, 0, y]
            xdata1 = geometry[0, 1, y]
            xdata2 = geometry[0, 2, y]
            xdata3 = geometry[0, 3, y]
            angles = geometry[0, 4, y]
        
            for x in range(cols):
        
                if scoresdata[x] < self.min_confidence:  # if score is less than min_confidence, ignore
                    continue
                offsetx = x * 4.0
                offsety = y * 4.0
                angle = angles[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = xdata0[x] + xdata2[x]
                w = xdata1[x] + xdata3[x]
                endx = int(offsetx + (cos * xdata1[x]) + (sin * xdata2[x]))
                endy = int(offsety + (sin * xdata1[x]) + (cos * xdata2[x]))
                startx = int(endx - w)
                starty = int(endy - h)
    
                rects.append((startx, starty, endx, endy))
                confidences.append(scoresdata[x])
        return (rects, confidences)


    def getText(self, fileName, image, fig_text):
        
        spellcorr = sc()
        imcpy = image.copy()
        (height, width) = image.shape[:2]
        (new_height, new_width) = (self.size, self.size) 
        im_resize = cv2.resize(image, (new_width, new_height))  # image2
        rW = width / float(new_width)
        rH = height / float(new_height)
        
        layerNames = [	"feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]     
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        blob = cv2.dnn.blobFromImage(im_resize, 1.0, (new_width, new_height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        (rects, confidences) = self.decodePredictions(scores, geometry)
        
        # applying non-maxima suppression to supppress weak and overlapping bounding boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences, overlapThresh=self.overlapTh)
        rectangle_list = self.boxToRec(boxes)
        # Merge rectangles
        merger = Merger()
        rectangle_list = merger.merge_rectangle_list(rectangle_list)
        
        
        results = []
        for rec in rectangle_list:
            startX = int(rec.x1 * rW)
            startY = int(rec.y1 * rH)
            endX = int(rec.x2 * rW)
            endY = int(rec.y2 * rH)
        
            dX = int((endX - startX) * self.padding)
            dY = int((endY - startY) * self.padding)
            
            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(width, endX + (dX * 2))
            endY = min(height, endY + (dY * 2))
     
        	# extract the actual padded ROI
            roi = imcpy[startY:endY, startX:endX]
            if (type(roi) is np.ndarray):             
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Apply preprocessing to enhance the text
            gray_roi = cv2.resize(gray_roi, (0,0), fx = 1.5, fy = 1.5)# interpolation = cv2.INTER_CUBIC)
            gray_roi = Image.fromarray(gray_roi)
            gray_roi = ImageEnhance.Sharpness(gray_roi).enhance(2)
            gray_roi = ImageEnhance.Contrast(gray_roi).enhance(2)
                
            output_dict = pytesseract.image_to_data(roi, config=self.tess_config, output_type='dict')
            # add the bounding box coordinates and OCR'd text to the list of results
            results.append(((startX, startY, endX, endY), output_dict))
            
        # sort the results bounding box coordinates from top to bottom
        results = sorted(results, key=lambda r:r[0][1])
        output = imcpy.copy()
        final_results = []
        # loop over the results
        for ((startX, startY, endX, endY), output_dict) in results:
        	   # display the text OCR'd by Tesseract
            text_list = output_dict.get('text')
            prob_list = output_dict.get('conf')
            text = ""
            prob = 0
            tot_text = 0
            for index in range(len(text_list)):
                
                op = re.sub(r'^[^()+*-<>#=&{}[\]/\0-9a-zA-Z]*', '', re.sub(r'[^()+*-<>#=&{}[\]/\0-9a-zA-Z]*$', '', text_list[index]))
                op = "".join([c if ord(c) < 128 else "" for c in op]).strip()
                op = re.sub('\W+','', op)
                if op != "":
                    op_cor = spellcorr.correctWord(op, fig_text)
                    #print("EAST op = %s, op_cor: %s"%(op, op_cor))
                    text = text+op_cor+" "
                    prob = (prob+prob_list[index])
                    tot_text = tot_text+1
                    
            if tot_text > 0:
                
                prob = prob/tot_text
                
                final_results.append(((startX, startY, endX, endY), text, prob))
                cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 1)
                cv2.putText(output, text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
         
        
        return final_results 
   
    

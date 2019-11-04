from __future__ import print_function
import numpy as np
import cv2
import pytesseract
from imutils.object_detection import non_max_suppression    # pip install imutils
from PIL import Image, ImageEnhance, ImageFilter
import re
from Rectangle import Rectangle
from RectangleMerger import RectangleMerger as Merger
from itertools import combinations 
import difflib
import os
from scipy import ndimage

import difflib
from SpellCorrect import SpellCorrect as sc
from TextDetectFullIm import TextDetectFullIm as td
from TextDetectEast import TextDetectEast as tde


class TextDetectAll:
    
    # def __init__(self):
    #     self.tess_config = ("-l eng --oem 1 --psm 11") ### Tesseract Confiuration for EAST Detector
    
    def getMergedText(self, pos1, text1, pos2, text2 ):
        
        if text1 != text2:
            d = difflib.Differ()
            diff = d.compare(text1, text2)
            
            merged_text = ""
            
            if pos1[0] < pos2[0] or pos1[1] < pos2[1] or pos1[2] < pos2[2] or pos1[3] < pos2[3] :
                merged_text += text1
                merged_text += text2
            else:
                merged_text += text2
                merged_text += text1
       
            return merged_text
        else:
            return text1
                  

    def getTextFullImage(self, fileName, im, fig_text):
        textdetector = td()
        config = '--psm 6 --oem 1'
        text_list = textdetector.getText(fileName, im, config, fig_text)
        return text_list
        
    def getTextEast(self, fileName, im, fig_text):
        textdetectoreast = tde()
        text_list = textdetectoreast.getText(fileName, im, fig_text)
        return text_list    
       
    def getTextComponent(self, fileName, components, image, fig_text):
#        textdetector = td()
        text_list = []
        textdetectoreast = tde()
        imcpy = image.copy()
        # get image ROI from component
        for c in components:
            # get bounding box
            x,y,w,h = cv2.boundingRect(c)
            get_roi = image[y:y+h, x:x+w]
            
            if (h > int(w*1.5)):
                get_roi = cv2.copyMakeBorder(get_roi,10,10,h,h,cv2.BORDER_CONSTANT,value=[255, 255, 255]) # add white border in the original image
                #get_roi = self.rotateImage(get_roi, -90) 
                get_roi = ndimage.rotate(get_roi, -90)
                
                text = textdetectoreast.getText(fileName, get_roi, fig_text)#textdetector.getText(fileName, get_roi, config)
                ftext = ""
                fprob = 0
                for ((startX, startY, endX, endY), op, prob) in text:
                    ftext += op
                    if prob>fprob:
                        fprob = prob
                        
                text_list.append(((x, y, x+w, y+h), ftext, fprob))
                cv2.rectangle(imcpy, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(imcpy, ftext, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#            else:
#                get_roi = cv2.copyMakeBorder(get_roi,10,10,10,10,cv2.BORDER_CONSTANT,value=[255, 255, 255]) # add white border in the original image
#                 
        
        return text_list
        
        
    def mergeOverlappingTextROI(self, text_list1, text_list2, fig_text):
        
        merge = Merger()
        merged_text_list = []
        spellcorr = sc()

        for ((startX1, startY1, endX1, endY1), op1, prob1) in text_list1:
            for ((startX2, startY2, endX2, endY2), op2, prob2) in text_list2:

                if merge._is_rectangles_overlapped_horizontally(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX2, startY2, endX2, endY2)):
                    rec = merge._merge_2_rectangles(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX2, startY2, endX2, endY2))
                    sop1 = op1.strip()
                    sop2 = op2.strip()
                    if (sop1==sop2):
                        op = sop1
                        if prob1 > prob2:
                            prob = prob1
                        else:
                            prob = prob2
                        merged_text_list.append(((rec.x1, rec.y1, rec.x2, rec.y2), op, prob))
                    elif re.match('^[0-9a-zA-Z]*$',sop1.lower().replace(" ", "")) and re.search(sop1.lower().replace(" ", ""), sop2.lower().replace(" ", ""), flags = 0) is not None:
                        op = sop2
                        prob = prob2
                        
                        merged_text_list.append(((rec.x1, rec.y1, rec.x2, rec.y2), op, prob))
                    elif re.match('^[0-9a-zA-Z]*$',sop2.lower().replace(" ", "")) and re.search(sop2.lower().replace(" ", ""), sop1.lower().replace(" ", ""), flags = 0) is not None:
                        op = sop1
                        prob = prob1
                        
                        merged_text_list.append(((rec.x1, rec.y1, rec.x2, rec.y2), op, prob))
                    else:

                        matcher = difflib.SequenceMatcher(None, sop1, sop2)
                        match = matcher.find_longest_match(0, len(sop1), 0, len(sop2))
                        if (match.size>1): 
                            if prob1 > prob2:
                                remaining_text = sop2[0:match.b] + sop2[match.b + match.size :]  
                                if len(remaining_text) > 1:
                                    remaining_text = spellcorr.correctWord(remaining_text, fig_text)
                                    op = sop1 + " " + remaining_text  
                                else:
                                    op = sop1

                                prob = prob1
                                merged_text_list.append(((rec.x1, rec.y1, rec.x2, rec.y2), op, prob))
                            else:
                                remaining_text = sop1[0:match.a] + sop1[match.a + match.size :] 
                                if len(remaining_text) > 1:
                                    remaining_text = spellcorr.correctWord(remaining_text, fig_text)
                                    op = sop2 + " " + remaining_text 
                                else:
                                    op = sop2
                                prob = prob2
                                merged_text_list.append(((rec.x1, rec.y1, rec.x2, rec.y2), op, prob))                   
                        else: 
                            if prob1>=50 and prob2>=50:
                                if prob1 > prob2:
                                    op = sop1 + " " + sop2  
                                    prob = prob1
                                else:
                                    op = sop2 + " " + sop1  
                                    prob = prob2
                                merged_text_list.append(((rec.x1, rec.y1, rec.x2, rec.y2), op, prob))                    
                            elif prob1>=50 and prob2<50:
                                op = sop1 
                                prob = prob1
                                merged_text_list.append(((rec.x1, rec.y1, rec.x2, rec.y2), op, prob))
                            elif prob2>=50 and prob1<50:
                                op = sop2 
                                prob = prob2
                             
                                merged_text_list.append(((rec.x1, rec.y1, rec.x2, rec.y2), op, prob))
                            else:
                                op = "NONE"  
                                prob = 0
                    #break
#                else:
#                    combined_results.append(((startX2, startY2, endX2, endY2), op2, prob2))                    
#                    combined_results.append(((startX1, startY1, endX1, endY1), op1, prob1))
                   
        return merged_text_list
         
    def combinedTextDetect(self, fileName, image, components, fig_text):
        
        final_text_im = image.copy()
        merge = Merger()
        combined_results = []
        clean_combined_results = []
        
        
        text_fullIm = self.getTextFullImage(fileName, image, fig_text)
        text_ROI = text_ROI = self.getTextEast(fileName, image, fig_text)
        text_componentIm = self.getTextComponent(fileName, components, image, fig_text)
        
        for ((startX1, startY1, endX1, endY1), op1, prob1) in text_componentIm:
            for ((startX2, startY2, endX2, endY2), op2, prob2) in text_fullIm:
                #print("Comparing %s within %s\n"% (op2, op1))             
             
                if merge._is_rectangles_inside(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX2, startY2, endX2, endY2)):
                    #print("removing op2 = %s"%(op2))                    
                    text_fullIm.remove(((startX2, startY2, endX2, endY2), op2, prob2))
                    break
                
        component_number = 0     
        for ((startX, startY, endX, endY), op, prob) in text_fullIm:
             #print("========")
             #print("Component %d: Position: (%d %d %d %d), text: %s, confidence = %d\n"% (component_number, startX, startY, endX, endY, op, prob))             
             component_number = component_number + 1 
             
        
        # sort the results bounding box coordinates from top to bottom
        text_fullIm = sorted(text_fullIm, key=lambda r:r[0][1])
        text_ROI = sorted(text_ROI, key=lambda r:r[0][1])
        text_componentIm = sorted(text_componentIm, key=lambda r:r[0][1])
        ################ Find if there are same text extracted from multiple detection methods, then keep only the best one #####################
        temp_combine = self.mergeOverlappingTextROI(text_fullIm, text_ROI, fig_text)
        combined_results = temp_combine.copy()
        combined_results = [combined_results[i] for i in range(len(combined_results)) if i == 0 or combined_results[i] != combined_results[i-1]]
        combined_results = sorted(combined_results, key=lambda r:r[0][1])
        comb = combinations(combined_results, 2)
        
        ################################### Check if there are horizontally near text boxes, then merge them ####################################
        
        for tup in list(comb):
            if (merge._is_rectangles_overlapped_horizontally(Rectangle.from_2_pos(tup[0][0][0],tup[0][0][1],tup[0][0][2],tup[0][0][3]), 
                                                            Rectangle.from_2_pos(tup[1][0][0],tup[1][0][1],tup[1][0][2],tup[1][0][3]))):
                                                                
                merged_rec_pos = merge._merge_2_rectangles(Rectangle.from_2_pos(tup[0][0][0],tup[0][0][1],tup[0][0][2],tup[0][0][3]), 
                                                 Rectangle.from_2_pos(tup[1][0][0],tup[1][0][1],tup[1][0][2],tup[1][0][3]))
                           
                
                if re.match('^[0-9a-zA-Z]*$',tup[0][1].lower().replace(" ", "")) and re.search(tup[0][1].lower().replace(" ", ""), tup[1][1].lower().replace(" ", ""), flags = 0) is not None:
                    merged_text = tup[1][1]
                    merged_conf = tup[1][2]
                        
                elif re.match('^[0-9a-zA-Z]*$',tup[1][1].lower().replace(" ", "")) and re.search(tup[1][1].lower().replace(" ", ""), tup[0][1].lower().replace(" ", ""), flags = 0) is not None:
                    merged_text = tup[0][1]
                    merged_conf = tup[0][2]
                        
                else:
                    merged_text = self.getMergedText(tup[0][0], tup[0][1], tup[1][0], tup[1][1]) #pos1, text1, pos2, text2
                    merged_conf = int((tup[0][2]+tup[1][2])/2)

                if tup[0] in combined_results:
                    combined_results.remove(tup[0])
                if tup[1] in combined_results:
                    combined_results.remove(tup[1])
                combined_results.append(((merged_rec_pos.x1, merged_rec_pos.y1, merged_rec_pos.x2, merged_rec_pos.y2), merged_text, merged_conf))
             
        ################################################# Include text detected only from EAST detector #######################################
        for ((startX2, startY2, endX2, endY2), op2, prob2) in text_ROI:
            find = 0
            for ((startX1, startY1, endX1, endY1), op1, prob1) in temp_combine:
                if merge._is_rectangles_overlapped_horizontally(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX2, startY2, endX2, endY2)):
                    find = 1
            if find == 0:
                combined_results.append(((startX2, startY2, endX2, endY2), op2, prob2))
        
        ############################################ Include text detected only from Full Image Tesseract detector #############################              
        for ((startX3, startY3, endX3, endY3), op3, prob3) in text_fullIm:
            find = 0
            for ((startX1, startY1, endX1, endY1), op1, prob1) in temp_combine:
                if merge._is_rectangles_overlapped_horizontally(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX3, startY3, endX3, endY3)):
                    find = 1
            if find == 0:
                combined_results.append(((startX3, startY3, endX3, endY3), op3, prob3))

        ############################################### Include text detected only from rotated components ######################################               
        for ((startX3, startY3, endX3, endY3), op3, prob3) in text_componentIm:
            find = 0
            for ((startX1, startY1, endX1, endY1), op1, prob1) in temp_combine:
                if merge._is_rectangles_overlapped_horizontally(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX3, startY3, endX3, endY3)):
                    find = 1
            if find == 0:
                combined_results.append(((startX3, startY3, endX3, endY3), op3, prob3))
                
                       
        combined_results = [combined_results[i] for i in range(len(combined_results)) if i == 0 or combined_results[i] != combined_results[i-1]]
        combined_results = sorted(combined_results, key=lambda r:r[0][1])
        
        ############################################ remove duplicate text boxes ###########################################
        for ((startX, startY, endX, endY), op, prob) in combined_results:
            find = 0
            for ((startX1, startY1, endX1, endY1), op1, prob1) in clean_combined_results:
                if (startX1, startY1, endX1, endY1) == (startX, startY, endX, endY) and op == op1:
                    find = 1
            if find == 0:
                clean_combined_results.append(((startX, startY, endX, endY), op, prob))
                
        component_number = 1
        for ((startX, startY, endX, endY), op, prob) in clean_combined_results:
             component_number = component_number + 1 
             cv2.rectangle(final_text_im, (startX, startY), (endX, endY), (0, 0, 255), 1)
             cv2.putText(final_text_im, op, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        clean_combined_results = sorted(clean_combined_results, key=lambda r:r[0][1])
        return clean_combined_results
        

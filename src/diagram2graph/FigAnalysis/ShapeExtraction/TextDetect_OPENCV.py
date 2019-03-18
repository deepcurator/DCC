from __future__ import print_function
import numpy as np
import cv2
import pytesseract
from imutils.object_detection import non_max_suppression    # pip install imutils
from PIL import Image, ImageEnhance, ImageFilter
from TextDetect import TextDetect as td
import re
from rectangle import Rectangle
from rectangle_merger import RectangleMerger as Merger
from itertools import combinations 
import difflib
import os

class TextDetect_EAST:
    
    def __init__(self):
        self.min_confidence = 0.3
        self.size = 320
        self.padding = 0.0
        self.overlapTh = 0.65
        self.tess_config = ("-l eng --oem 1 --psm 11")
    
    def get_merged_text(self, pos1, text1, pos2, text2, ):
        
        if text1 != text2:
            d = difflib.Differ()
            diff = d.compare(text1, text2)
            #print('\n'.join(list(diff)))
            
            merged_text = ""
            
            if pos1[0] < pos2[0] or pos1[1] < pos2[1] or pos1[2] < pos2[2] or pos1[3] < pos2[3] :
                merged_text += text1
                merged_text += text2
            else:
                merged_text += text2
                merged_text += text1
#        line_no = 0        
#        for line in diff:
#            print(line)
#            code = line[:2]
#            print(code)
#            if code in (" ", "+ "):
#                line_no +=1
#            if code == "+ ":
#                print("%d: %s" % (line_no, line[2:].strip()))
#                merged_text += code
            return merged_text
        else:
            return text1
                
       
    def boxToRec(self, box):
        rec = []
        for x1, y1, x2, y2, in box:
            r = Rectangle.from_2_pos(x1, y1, x2, y2)
            rec.append(r)
        return rec  
                   
    def decode_predictions(self, scores, geometry):
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


    def getTextROI(self, fileName, image):
        
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
        (rects, confidences) = self.decode_predictions(scores, geometry)
        
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
            
                #gray_roi = cv2.GaussianBlur(gray_roi, (3,3), 0)
                #gray_roi = cv2.bilateralFilter(gray_roi, 9, 25, 25)
                #gray_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                #gray_roi = cv2.adaptiveThreshold(gray_roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            gray_roi = cv2.resize(gray_roi, (0,0), fx = 1.5, fy = 1.5)# interpolation = cv2.INTER_CUBIC)
            gray_roi = Image.fromarray(gray_roi)
            gray_roi = ImageEnhance.Sharpness(gray_roi).enhance(2)
            gray_roi = ImageEnhance.Contrast(gray_roi).enhance(2)
                
            #text = pytesseract.image_to_string(roi, config=self.tess_config)
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
            #print("========")
            #print("Before: {}\n".format(text_list))
            #print("Before conf: {}\n".format(prob_list))
            text = ""
            prob = 0
            tot_text = 0
            for index in range(len(text_list)):
                #print(text_list[index])
                op = re.sub(r'^[^()+*-<>#={}[\]/\0-9a-zA-Z]*', '', re.sub(r'[^()+*-<>#={}[\]/\0-9a-zA-Z]*$', '', text_list[index]))
                op = "".join([c if ord(c) < 128 else "" for c in op]).strip()
                op = re.sub('\W+','', op)
                if op != "":
                    text = text+op+" "
                    prob = (prob+prob_list[index])
                    tot_text = tot_text+1
                    #print("Text: {}\n".format(text))
            #print("After: {}\n".format(text))
            if tot_text > 0:
                #print("After conf: {}\n".format(prob/tot_text))
                prob = prob/tot_text
                final_results.append(((startX, startY, endX, endY), text, prob))
                cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 1)
                cv2.putText(output, text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
         
        # show the output image
#        cv2.imshow("Text Detect East", output)
#        cv2.waitKey(0)
#        op_dir = "/home/z003yk0t/DiagramAnalysis/ContentExtraction/ShapeExtraction/2Dresults_validation/TextEast"
#        op_file_name = os.path.join(op_dir, "op" + os.path.basename(fileName))
#        print(op_file_name)
#        cv2.imwrite(op_file_name, output)
            
        return final_results 
        
    def getTextFullImage(self, fileName, im):
        textdetector = td()
        text_list = textdetector.getText(fileName, im)
        return text_list
            
    def combinedTextDetect(self, fileName, image):
        
        final_text_im = image.copy()
        
        text_fullIm = self.getTextFullImage(fileName, image)
        text_ROI = self.getTextROI(fileName, image)
        
        # sort the results bounding box coordinates from top to bottom
        text_fullIm = sorted(text_fullIm, key=lambda r:r[0][1])
        text_ROI = sorted(text_ROI, key=lambda r:r[0][1])
        merge = Merger()
        combined_results = []
        temp_combine = []
        for ((startX1, startY1, endX1, endY1), op1, prob1) in text_fullIm:
            for ((startX2, startY2, endX2, endY2), op2, prob2) in text_ROI:
#                print("========")
#                print("text_fullIm op1: {}\n".format(op1))
#                print("text_ROI op2: {}\n".format(op2))
                if merge._is_rectangles_overlapped_horizontally(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX2, startY2, endX2, endY2)):
                    rec = merge._merge_2_rectangles(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX2, startY2, endX2, endY2))
                    if re.search(op1, op2, flags = 0) is not None:
                        op = op2
                        prob = prob2
                        #print("=== op1 in op2 ===")
                    elif re.search(op2, op1, flags = 0) is not None:
                        op = op1
                        prob = prob1
                        #print("=== op2 in op1 ===")
                    else:
                         if prob1 > prob2:
                             op = op1
                             prob = prob1
                             #print("=== op1 & op2 different, prob1 > prob2 ===")
                         else:
                             op = op2
                             prob = prob2
                             #print("=== op1 & op2 different, prob2 > prob1 ===")
                    temp_combine.append(((rec.x1, rec.y1, rec.x2, rec.y2), op, prob))
#                    cv2.rectangle(final_text_im, (rec.x1, rec.y1), (rec.x2, rec.y2), (0, 0, 255), 1)
#                    cv2.putText(final_text_im, op, (rec.x1, rec.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                    cv2.imshow("final_text_im", final_text_im)
#                    cv2.waitKey(0)
                    break
#                else:
#                    combined_results.append(((startX2, startY2, endX2, endY2), op2, prob2))                    
#                    combined_results.append(((startX1, startY1, endX1, endY1), op1, prob1))
                    #print("=== does not satisfy any condition ===")
        combined_results = temp_combine.copy()
        combined_results = [combined_results[i] for i in range(len(combined_results)) if i == 0 or combined_results[i] != combined_results[i-1]]
        combined_results = sorted(combined_results, key=lambda r:r[0][1])
        comb = combinations(combined_results, 2)
        for tup in list(comb):
            if (merge._is_rectangles_overlapped_horizontally(Rectangle.from_2_pos(tup[0][0][0],tup[0][0][1],tup[0][0][2],tup[0][0][3]), 
                                                            Rectangle.from_2_pos(tup[1][0][0],tup[1][0][1],tup[1][0][2],tup[1][0][3]))):
#                print("********************")
#                print("Merging Component 1: (%d %d %d %d), %s, %d, Component 2: (%d %d %d %d), %s, %d\n"% (tup[0][0][0],tup[0][0][1],tup[0][0][2],tup[0][0][3], tup[0][1], tup[0][2],
#                                                                                 tup[1][0][0],tup[1][0][1],tup[1][0][2],tup[1][0][3], tup[1][1], tup[1][2])) 
                                                                                 
                merged_rec_pos = merge._merge_2_rectangles(Rectangle.from_2_pos(tup[0][0][0],tup[0][0][1],tup[0][0][2],tup[0][0][3]), 
                                                 Rectangle.from_2_pos(tup[1][0][0],tup[1][0][1],tup[1][0][2],tup[1][0][3]))
                merged_text = self.get_merged_text(tup[0][0], tup[0][1], tup[1][0], tup[1][1])
                merged_conf = int((tup[0][2]+tup[1][2])/2)
#                print("Merged Rec Position: (%d %d %d %d), text = %s, conf = %d \n"% (merged_rec_pos.x1, merged_rec_pos.y1, merged_rec_pos.x2, merged_rec_pos.y2, merged_text, merged_conf))    
                if tup[0] in combined_results:
                    combined_results.remove(tup[0])
                if tup[1] in combined_results:
                    combined_results.remove(tup[1])
                combined_results.append(((merged_rec_pos.x1, merged_rec_pos.y1, merged_rec_pos.x2, merged_rec_pos.y2), merged_text, merged_conf))
             
#        component_number = 1
#        for ((startX, startY, endX, endY), op, prob) in combined_results:
#             print("========")
#             print("Component %d: Position: (%d %d %d %d), text: %s, confidence = %d\n"% (component_number, startX, startY, endX, endY, op, prob))             
#             component_number = component_number + 1 
#             cv2.rectangle(final_text_im, (startX, startY), (endX, endY), (0, 255, 0), 2)
#             
#             cv2.putText(final_text_im, op, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#        cv2.imshow("final_text_im", final_text_im)
#        cv2.waitKey(0)
        
        for ((startX2, startY2, endX2, endY2), op2, prob2) in text_ROI:
            find = 0
            for ((startX1, startY1, endX1, endY1), op1, prob1) in temp_combine:
                if merge._is_rectangles_overlapped_horizontally(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX2, startY2, endX2, endY2)):
                    find = 1
            if find == 0:
                combined_results.append(((startX2, startY2, endX2, endY2), op2, prob2))
               
        for ((startX3, startY3, endX3, endY3), op3, prob3) in text_fullIm:
            find = 0
            for ((startX1, startY1, endX1, endY1), op1, prob1) in temp_combine:
                if merge._is_rectangles_overlapped_horizontally(Rectangle.from_2_pos(startX1, startY1, endX1, endY1), Rectangle.from_2_pos(startX3, startY3, endX3, endY3)):
                    find = 1
            if find == 0:
                combined_results.append(((startX3, startY3, endX3, endY3), op3, prob3))
                
                     
        combined_results = [combined_results[i] for i in range(len(combined_results)) if i == 0 or combined_results[i] != combined_results[i-1]]
        combined_results = sorted(combined_results, key=lambda r:r[0][1])
        
        component_number = 1
        for ((startX, startY, endX, endY), op, prob) in combined_results:
             print("========")
             print("Component %d: Position: (%d %d %d %d), text: %s, conf = %d \n"% (component_number, startX, startY, endX, endY, op, prob))             
             component_number = component_number + 1 
             cv2.rectangle(final_text_im, (startX, startY), (endX, endY), (0, 0, 255), 1)
             cv2.putText(final_text_im, op, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # show the output image
#        cv2.imshow("final_text_im", final_text_im)
#        cv2.waitKey(0)
#        op_dir = "/home/z003yk0t/DiagramAnalysis/ContentExtraction/ShapeExtraction/2Dresults_validation/TextCombined"
##        op_file_name = os.path.join(op_dir, "op" + os.path.basename(fileName))
##        print(op_file_name)
#        cv2.imwrite(op_file_name, final_text_im)
        return combined_results
        
from __future__ import print_function
import numpy as np
import cv2
import math
from scipy import ndimage
from LineMerger import LineMerger as LMerger


class ArrowDetect:
    
    def __init__(self):
        
        self.overlap_dist = -7
        self.min_line_length = 2
        self.max_dist_init = -1000000
        self.min_dist_thresh = 25

   
    def findNeighborPixel(self, extendX, extendY, otherendX, otherendY, thresh_imcpy):
        (height, width) = thresh_imcpy.shape[:2]
        diffx = np.sign(extendX - otherendX)
        diffy = np.sign(extendY - otherendY)
        window_sizeX = 5*diffx
        window_sizeY = 5*diffy
        listX = list(range(min(window_sizeX, 0), max(window_sizeX, 0)))
        listY = list(range(min(window_sizeY, 0), max(window_sizeY, 0)))
      
        for i in listX:
            for j in listY:
                posX = extendX + i
                posY = extendY + j
                if (i != 0 or j != 0):                  
                    if posX <= width and posY <= height:
                        if (thresh_imcpy[posY][posX] == 255):
                            return(posX, posY)
        return (extendX, extendY)
      
    def reachComponentBox(self, components, px, py):
        for index in range(len(components)):   
            dist = cv2.pointPolygonTest(components[index],(px, py),True)             
            if dist >= -5:
                return (True, index) 
        return (False, -1)
    
    
    def reachTextBox(self, text_list, px, py):
        for index in range(len(text_list)):
            ((startX, startY, endX, endY), op, prob) = text_list[index]
            comp = np.array([[[startX, startY]],[[startX, endY]],[[endX, endY]], [[endX, startY]]], dtype=np.int32)                              
            dist = cv2.pointPolygonTest(comp,(px, py),False)
            if dist >= 0:
                return (True, index) 
        return (False, -1)

    def findClosestTextPoint(self, px, py, text_list, exclude_comp):
        max_dist = self.max_dist_init 
        nearest_comp = -1
        
        for index in range(len(text_list)):
            if(index != exclude_comp):
                ((startX, startY, endX, endY), op, prob) = text_list[index]
                comp = np.array([[[startX, startY]],[[endX, startY]],[[endX, endY]],[[startX, endY]]], dtype=np.int32)
                D = cv2.pointPolygonTest(comp,(px, py),True)
                if D > max_dist:
                    max_dist = D
                    nearest_comp = index
        return (nearest_comp, abs(max_dist))   
        
    def findClosestComponentPoint(self, px, py, components, exclude_comp):
        max_dist = self.max_dist_init 
        nearest_comp = -1
        
        for index in range(len(components)): 
            if(index != exclude_comp):
                D = cv2.pointPolygonTest(components[index],(px, py),True)             
                if D > max_dist:
                    max_dist = D
                    nearest_comp = index
        return (nearest_comp, abs(max_dist))
        
        
    def extendLine(self, lines, img, thresh_im, components, text_list):
        
        imcpy = img.copy()
        (height, width) = img.shape[:2]
        thresh_imcpy = thresh_im.copy()
        kernel_dil = np.ones((2,2),np.uint8)
        thresh_imcpy = cv2.dilate(thresh_imcpy,kernel_dil,iterations = 2) 
        thresh_imcpy = cv2.erode(thresh_imcpy,kernel_dil,iterations = 1) 
        
        extended_lines = []
        lmerger = LMerger()
        
        while lines: 
            line = lines.pop(0)
            extend_line_x1 = -1
            extend_line_y1 = -1
            extend_line_x2 = -1
            extend_line_y2 = -1
            still_extending = True
            start_comp_found = -1
            end_comp_found = -1
            start_text_found = -1
            end_text_found = -1
            for x1, y1, x2, y2 in line:
                if(extend_line_x1 == -1 and extend_line_y1 == -1 and extend_line_x2 == -1 and extend_line_y2 == -1):
                    extend_line_x1 = x1
                    extend_line_y1 = y1
                    extend_line_x2 = x2
                    extend_line_y2 = y2
                    
                while(still_extending):
                    # check if the start of the line already reached to a component or edge  
                    if start_comp_found == -1 and start_text_found == -1 and (extend_line_x1 >0 and extend_line_x1 < width and extend_line_y1 >0 and extend_line_y1< height):
                        startx, starty = self.findNeighborPixel(extend_line_x1, extend_line_y1, extend_line_x2, extend_line_y2, thresh_imcpy)
                        # reached to another line
                        isMergable, lineID, mergex, mergey  = lmerger.isMergableLine(lines, startx, starty)
                        if (isMergable):                                               
                            startx = mergex
                            starty = mergey
                            lines.pop(lineID)
                        # reached to a component box 
                        reachComp, component_index = self.reachComponentBox(components, startx, starty)
                        if(reachComp):
                            start_comp_found = component_index
                        # reached to a text box
                        reachText, text_index = self.reachTextBox(text_list, startx, starty)
                        if(reachText):
                            start_text_found == text_index
                    else:
                        startx = extend_line_x1
                        starty = extend_line_y1

                    # check if the end of the line already reached to a component or edge     
                    if end_comp_found == -1 and end_text_found == -1 and (extend_line_x2 >0 and extend_line_x2 < width and extend_line_y2 >0 and extend_line_y2< height):
                        endx, endy = self.findNeighborPixel(extend_line_x2, extend_line_y2, extend_line_x1, extend_line_y1, thresh_imcpy)
                        # reached to another line
                        isMergable, lineID, mergex, mergey  = lmerger.isMergableLine(lines, endx, endy)
                        if (isMergable):                                               
                            endx = mergex
                            endy = mergey
                            lines.pop(lineID)
                        # reached to a component box 
                        reachComp, component_index = self.reachComponentBox(components, endx, endy)
                        if(reachComp):
                            end_comp_found = component_index
                        # reached to a text box
                        reachText, text_index = self.reachTextBox(text_list, endx, endy)
                        if(reachText):
                            end_text_found == text_index
                    else:
                        endx = extend_line_x2
                        endy = extend_line_y2
                        
                    # reached at the deadend without any of the above or found components
                    if(extend_line_x1 == startx and extend_line_y1 == starty and extend_line_x2 == endx and extend_line_y2 == endy):
                        still_extending = False
                        extended_lines.append(((startx, starty, endx, endy), start_comp_found, end_comp_found, start_text_found, end_text_found))    
                        cv2.line(imcpy, (startx, starty), (endx, endy), (255,0,255), 2)
                        
                    # still extending
                    else:
                        extend_line_x1 = startx
                        extend_line_y1 = starty
                        extend_line_x2 = endx
                        extend_line_y2 = endy
                    
        # cv2.imshow('arrow_detect imcpy',imcpy)
        # cv2.waitKey(0)
    
        return extended_lines
        
        
    def discardLineInsideComp(self, lines, components, text_list):
        discarded_lines = [] 
        retained_lines = []
        
        for line in lines:
            flag = 0
            dist1 = -10000;
            dist2 = -10000;
            
            for x1, y1, x2, y2 in line:                              
                for cp in components:                
                    dist1 = cv2.pointPolygonTest(cp,(x1, y1),True) 
                    dist2 = cv2.pointPolygonTest(cp,(x2, y2),True)
                    
                    if dist1 >= self.overlap_dist and dist2 >= self.overlap_dist:
                        flag = 1                  
                        
                if flag == 0:
                    for ((startX, startY, endX, endY), op, prob) in text_list:             
                        if (x1>=startX and x1<=endX and y1>=startY and y1<=endY) and (x2>=startX and x2<=endX and y2>=startY and y2<=endY) :
                            flag = 1 
                
            if flag == 1:
                discarded_lines.append([[x1, y1, x2, y2]])
            else:
                retained_lines.append([[x1, y1, x2, y2]])
                        
        retained_lines = sorted(retained_lines)
        retained_lines = [retained_lines[i] for i in range(len(retained_lines)) if i == 0 or retained_lines[i] != retained_lines[i-1]]
        
        return retained_lines
    
    
    def getLineCompTag(self, img, thresh_im, components, retained_lines, text_list):

        line_list = []
        extended_lines = self.extendLine(retained_lines, img, thresh_im, components, text_list)
        
        for ((startx, starty, endx, endy), start_comp_found, end_comp_found, start_text_found, end_text_found) in extended_lines:
            
            if(start_comp_found == -1 and end_comp_found != -1):
                neighbor_comp_index_start, neighbor_comp_dist_start = self.findClosestComponentPoint(startx, starty, components, end_comp_found)
                if neighbor_comp_dist_start < self.min_dist_thresh: 
                    start_comp_found = neighbor_comp_index_start #update start_comp_found  
            elif(start_comp_found != -1 and end_comp_found == -1):
                neighbor_comp_index_end, neighbor_comp_dist_end = self.findClosestComponentPoint(endx, endy, components, start_comp_found)
                if neighbor_comp_dist_end < self.min_dist_thresh: 
                    end_comp_found = neighbor_comp_index_end #update end_comp_found 
                    
            if start_comp_found == -1:
                neighbor_comp_index_start, neighbor_comp_dist_start = self.findClosestComponentPoint(startx, starty, components, -1)
                if neighbor_comp_dist_start < self.min_dist_thresh: 
                    start_comp_found = neighbor_comp_index_start #update start_comp_found 
            if start_text_found == -1 and start_comp_found == -1:
                neighbor_text_index_start, neighbor_text_dist_start = self.findClosestTextPoint(startx, starty, text_list, -1)
                if neighbor_text_dist_start < self.min_dist_thresh: 
                    start_text_found = neighbor_text_index_start #update start_text_found 
            
            if end_comp_found == -1:
                neighbor_comp_index_end, neighbor_comp_dist_end = self.findClosestComponentPoint(endx, endy, components, -1)
                if neighbor_comp_dist_end < self.min_dist_thresh: 
                    end_comp_found = neighbor_comp_index_end #update end_comp_found 
            if end_text_found == -1 and end_comp_found == -1:
                neighbor_text_index_end, neighbor_text_dist_end = self.findClosestTextPoint(startx, starty, text_list, -1)
                if neighbor_text_dist_end < self.min_dist_thresh: 
                    end_text_found = neighbor_text_index_end #update start_text_found 
   
            if (start_comp_found != -1 or start_text_found != -1) and (end_comp_found != -1 or end_text_found != -1):
                if(start_comp_found != end_comp_found) or (start_text_found != end_text_found):
                
                    if line_list == []:
                        line_list.append(((startx, starty, endx, endy), start_comp_found, end_comp_found, start_text_found, end_text_found))  
                          
                    else:   
                        already_exist = False
                        for((sx, sy, ex, ey), scf, ecf, stf, etf) in line_list:
                           comp_connect = [scf, ecf]
                           text_connect = [stf, etf]
                           if (start_comp_found in comp_connect and end_comp_found in comp_connect and start_comp_found != -1 and end_comp_found != -1) or (start_text_found in text_connect and end_text_found in text_connect and start_text_found != -1 and end_text_found != -1):
                                already_exist = True
         
                        if already_exist == False:
                            line_list.append(((startx, starty, endx, endy), start_comp_found, end_comp_found, start_text_found, end_text_found))    
         
        return line_list
        
        
        
    def detectLines(self, img, thresh_im, gray_imcv, components, text_list):
    
        thresh_imcpy = thresh_im.copy()
        kernel_dil = np.ones((2,2),np.uint8)
        kernel_er = np.ones((5,5),np.uint8)
        thresh_imcpy = cv2.dilate(thresh_imcpy,kernel_dil,iterations = 1)        
        thresh_imcpy = cv2.erode(thresh_imcpy,kernel_er,iterations = 1)
        thresh_imcpy = thresh_im - thresh_imcpy
        #perform HoughLines on the image
        lines = cv2.HoughLinesP(thresh_imcpy, rho=1, theta=np.pi/180, threshold=15, minLineLength = self.min_line_length, maxLineGap = 0)        
        retained_lines = self.discardLineInsideComp(lines, components, text_list)
        
        # imcpy2 = img.copy()
        # for line in retained_lines:
        #     for x1, y1, x2, y2 in line:
        #         cv2.line(imcpy2,(x1,y1),(x2,y2),(255,0,0),2)
                
        # cv2.imshow('img_copy2', imcpy2)
        # cv2.waitKey(0)
        
        return retained_lines
            

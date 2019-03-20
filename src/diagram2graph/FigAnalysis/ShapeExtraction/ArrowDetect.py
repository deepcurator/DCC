from __future__ import print_function
import numpy as np
import cv2
import math
from scipy import ndimage

class ArrowDetect:
    
    def __init__(self):
        self.min_distance_to_merge = 10
        self.min_angle_to_merge = 1
        self.overlap_dist = -7
        self.min_line_length = 2

    def get_lines(self, lines_in):
        if cv2.__version__ < '3.0':
            return lines_in[0]
        return [l[0] for l in lines_in]
    
    
    def get_distance(self,line1, line2):
        dist1 = self.DistancePointLine(line1[0][0], line1[0][1], 
                                  line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist2 = self.DistancePointLine(line1[1][0], line1[1][1], 
                                  line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        dist3 = self.DistancePointLine(line2[0][0], line2[0][1], 
                                  line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        dist4 = self.DistancePointLine(line2[1][0], line2[1][1], 
                                  line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    
    
        return min(dist1,dist2,dist3,dist4)
    
    
    
    def merge_lines_pipeline(self, lines):
        super_lines_final = []
        super_lines = []
    
        for line in lines:
            create_new_group = True
            group_updated = False
    
            for group in super_lines:
                for line2 in group:
                    if self.get_distance(line2, line) < self.min_distance_to_merge:
                        # check the angle between lines       
                        orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                        orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))
    
                        if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < self.min_angle_to_merge: 
                            #print("angles", orientation_i, orientation_j)
                            #print(int(abs(orientation_i - orientation_j)))
                            group.append(line)
    
                            create_new_group = False
                            group_updated = True
                            break
    
                if group_updated:
                    break
    
            if (create_new_group):
                new_group = []
                new_group.append(line)
    
                for idx, line2 in enumerate(lines):
                    # check the distance between lines
                    if self.get_distance(line2, line) < self.min_distance_to_merge:
                        # check the angle between lines       
                        orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                        orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))
    
                        if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < self.min_angle_to_merge: 
                            #print("angles", orientation_i, orientation_j)
                            #print(int(abs(orientation_i - orientation_j)))
    
                            new_group.append(line2)
    
                            # remove line from lines list
                            #lines[idx] = False
                # append new group
                super_lines.append(new_group)


        for group in super_lines:
            super_lines_final.append(self.merge_lines_segments(group))
    
        return super_lines_final
    
    
    def lineMagnitude(self, x1, y1, x2, y2):
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
        return lineMagnitude
        


    def DistancePointLine(self, px, py, x1, y1, x2, y2):
        #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        LineMag = self.lineMagnitude(x1, y1, x2, y2)
    
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine
    
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)
    
        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = self.lineMagnitude(px, py, x1, y1)
            iy = self.lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = self.lineMagnitude(px, py, ix, iy)
    
        return DistancePointLine

    
    def merge_lines_segments(self,lines, use_log=False):
        if(len(lines) == 1):
            return lines[0]
    
        line_i = lines[0]
    
        # orientation
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
    
        points = []
        for line in lines:
            points.append(line[0])
            points.append(line[1])
    
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < 135:
    
            #sort by y
            points = sorted(points, key=lambda point: point[1])
    
            if use_log:
                print("use y")
        else:
    
            #sort by x
            points = sorted(points, key=lambda point: point[0])
    
            if use_log:
                print("use x")
    
        return [points[0], points[len(points)-1]]
    
    def entendLine(self, lines, img, thresh_im, gray_imcv, components, text_list):
        
        imcpy = img.copy()
        (height, width) = img.shape[:2]
        thresh_imcpy = thresh_im.copy()
        kernel_dil = np.ones((2,2),np.uint8)
        thresh_imcpy = cv2.dilate(thresh_imcpy,kernel_dil,iterations = 2)        
        cv2.imshow("thresh_imcpy", thresh_imcpy)
        cv2.waitKey(0)
        extended_lines = []
        while lines: #for line in lines:
            line = lines.pop(0)
            extend_line_x1 = 0
            extend_line_y1 = 0
            extend_line_x2 = 0
            extend_line_y2 = 0
            still_extending = 1
            start_comp_found = -1
            end_comp_found = -1
            start_text_found = -1
            end_text_found = -1
            for x1, y1, x2, y2 in line:
                if(extend_line_x1 == -1 and extend_line_y1 == -1 and extend_line_x2 == 0 and extend_line_y2 == 0):
                    extend_line_x1 = x1
                    extend_line_y1 = y1
                    extend_line_x2 = x2
                    extend_line_y2 = y2
                while(still_extending):
                    # check if already reached to a component or edge and 
                    if start_comp_found == -1 and start_text_found == -1 and (extend_line_x1 >0 and extend_line_x1 < width and extend_line_y1 >0 and extend_line_y1< height):
                        startx, starty = self.findNeighborPixel(extend_line_x1, extend_line_y1, thresh_imcpy)
                         # reached to another line
                        isMergable, lineID, mergex, mergey  = self.isMergableLine(lines, startx, starty)
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
                    if end_comp_found == 0 and end_text_found == 0 and (extend_line_x2 >0 and extend_line_x2 < width and extend_line_y2 >0 and extend_line_y2< height):
                        endx, endy = self.findNeighborPixel(extend_line_x2, extend_line_y2, thresh_imcpy)
                        # reached to another line
                        isMergable, lineID, mergex, mergey  = self.isMergableLine(lines, endx, endy)
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
                    
                    # reached at the deadend without any of the above
                    if(extend_line_x1 == startx and extend_line_y1 == starty and extend_line_x2 == endx and extend_line_y2 == endy):
                        still_extending = 0
                        extended_lines.append(((startx, starty, endx, endy), start_comp_found, end_comp_found, start_text_found, end_text_found))                           
                       
                    else:
                        extend_line_x1 = startx
                        extend_line_y1 = starty
                        extend_line_x2 = endx
                        extend_line_y2 = endy
                    
                
        return extended_lines
    
    def detectLines(self, img, thresh_im, gray_imcv, components, text_list):
    
        imcpy = img.copy()
        thresh_imcpy = thresh_im.copy()
        kernel_dil = np.ones((2,2),np.uint8)
        kernel_er = np.ones((5,5),np.uint8)
        thresh_imcpy = cv2.dilate(thresh_imcpy,kernel_dil,iterations = 1)        
        thresh_imcpy = cv2.erode(thresh_imcpy,kernel_er,iterations = 1)
#        D = ndimage.distance_transform_edt(thresh_imcpy)
#        cv2.imshow("D", D)
#        cv2.waitKey(0)
        #gray_imcvcpy = gray_imcv.copy()
        thresh_imcpy = thresh_im - thresh_imcpy
#        cv2.imshow("thresh_imcpy", thresh_imcpy)
#        cv2.waitKey(0)
        #show what the image looks like after the application of previous functions
            
#        edges = cv2.Canny(thresh_imcpy,threshold1=50,threshold2=200,apertureSize = 3)    
#        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
#        
#        cv2.imshow("canny's image", edges)
#        cv2.waitKey(0)
            
        #perform HoughLines on the image
        lines = cv2.HoughLinesP(thresh_imcpy, rho=1, theta=np.pi/180, threshold=15, minLineLength = self.min_line_length, maxLineGap = 0)
        print(lines.shape)
        discarded_lines = [] 
        retained_lines = []
        #print("-----------------------------total line before: %d ----------------------------- \n" % len(lines))
        img_copy1 = img.copy()
        img_copy2 = img.copy()
        
        for line in lines:
            flag = 0
            dist1 = -10000;
            dist2 = -10000;
            
            for x1, y1, x2, y2 in line: 
                #print(x1, y1, x2, y2, line)                
                for cp in components:                
                    dist1 = cv2.pointPolygonTest(cp,(x1, y1),True) 
                    dist2 = cv2.pointPolygonTest(cp,(x2, y2),True)
                    
                    if dist1 >= self.overlap_dist and dist2 >= self.overlap_dist:
                        flag = 1                  
                        #print("dist1 = %d, dist2 = %d " % (dist1, dist2))
                        cv2.drawContours(img_copy1, [cp], 0, (0, 255, 0), 2)
                        cv2.line(img_copy1,(x1,y1),(x2,y2),(0, 255, 0),2)            
                      
                
                if flag == 0:
                    for ((startX, startY, endX, endY), op, prob) in text_list:             
                        if (x1>=startX and x1<=endX and y1>=startY and y1<=endY) and (x2>=startX and x2<=endX and y2>=startY and y2<=endY) :
                            flag = 1 
                            cv2.line(img_copy1,(x1,y1),(x2,y2),(0,0,255),2) 
                            cv2.rectangle(img_copy1, (startX, startY), (endX, endY), (0, 0, 255), 2)
                                    
                
                
            if flag == 1:
                discarded_lines.append([[x1, y1, x2, y2]])
            else:
                retained_lines.append([[x1, y1, x2, y2]])
                        
        retained_lines = sorted(retained_lines)
        retained_lines = [retained_lines[i] for i in range(len(retained_lines)) if i == 0 or retained_lines[i] != retained_lines[i-1]]
        
#        print("--------------------------------total discarded lines: %d -------------------------- \n" % len(discarded_lines))
#        print("--------------------------------total retained lines: %d -------------------------- \n" % len(retained_lines))
        
        for line in retained_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_copy2,(x1,y1),(x2,y2),(255,0,0),2)
                
       
        cv2.imshow('img_copy1', img_copy1)
        cv2.waitKey(0) 
        cv2.imshow('img_copy2', img_copy2)
        cv2.waitKey(0)
        
        extended_lines = self.entendLine(retained_lines, img, thresh_im, gray_imcv, components, text_list)
        # merge lines
    
        #------------------
        # prepare
#        _lines = []
#        for _line in self.get_lines(retained_lines):
#            _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])
#    
#        # sort
#        _lines_x = []
#        _lines_y = []
#        for line_i in _lines:
#            orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
#            if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < 135:
#                _lines_y.append(line_i)
#            else:
#                _lines_x.append(line_i)
#    
#        _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
#        _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])
#    
#        merged_lines_x = self.merge_lines_pipeline(_lines_x)
#        merged_lines_y = self.merge_lines_pipeline(_lines_y)
#    
        merged_lines_all = []
#        merged_lines_all.extend(merged_lines_x)
#        merged_lines_all.extend(merged_lines_y)
##        print("process groups lines", len(_lines), len(merged_lines_all))
#        
#        for line in merged_lines_all:
#            cv2.line(imcpy, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 2)
#    
#    
#        cv2.imshow('arrow_detect',imcpy)
#        cv2.waitKey(0)
        return merged_lines_all
            

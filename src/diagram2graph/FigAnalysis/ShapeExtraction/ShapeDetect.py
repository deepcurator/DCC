from __future__ import print_function
import numpy as np
import imutils
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import os
from rectangle import Rectangle
from rectangle_merger import RectangleMerger as Merger



class ShapeDetect:
    
    def __init__(self):
        self.min_component_area = 700
        self.max_component_area = 25000
        self.overlapTh = 0.50
        

    def convert_contour2box(self, cnt):
        box = []
        for c in cnt:
            x,y,w,h = cv2.boundingRect(c)
            box.append([x, y, x+w, y+h])
        return np.array(box)
    
    def convert_contour2rect(self, cnt):
        rect = []
        for c in cnt:
            x,y,w,h = cv2.boundingRect(c)
            rect.append(Rectangle.from_2_pos(x, y, x+w, y+h))
        return np.array(rect)
        
    def get_merge_index(self, component, c):
        merger = Merger()
        rects = self.convert_contour2rect(component)
        x, y, w, h = cv2.boundingRect(c)
        index = []
        for i in range(0, len(rects)):
            if merger._is_rectangles_overlapped_horizontally(rects[i], Rectangle.from_2_pos(x, y, x+w, y+h)):
                index.append(i)
                
        return index
    
    def mergeOrAppend(self, c, shape, component, conf, imcpy, merge = 0):
        hull = cv2.convexHull(c)                 
        x, y, width, height = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        hullarea = cv2.contourArea(hull)
        solidity = float(area)/float(hullarea)
        
        merger = Merger()  
        mergable_index = []
        if merge == 1:
            mergable_index = self.get_merge_index(component, c)
        if solidity > 0.4:
            if len(mergable_index) != 0:
                
                for ind in mergable_index:
                    #print("ind = %d, Current Detected component is a %s\n"%(ind, shape))
                    mergable_conponent = component[ind]
                    x1, y1, w1, h1 = cv2.boundingRect(c)
                    x2, y2, w2, h2 = cv2.boundingRect(mergable_conponent)
                    merge_rec = merger._merge_2_rectangles(Rectangle.from_2_pos(x1, y1, x1+w1, y1+h1), Rectangle.from_2_pos(x2, y2, x2+w2, y2+h2))
                    merge_contour = np.array([[[merge_rec.x1, merge_rec.y1]],[[merge_rec.x2, merge_rec.y1]],[[merge_rec.x2, merge_rec.y2]], [[merge_rec.x1, merge_rec.y2]]], dtype=np.int32)
                   
                    component.pop(ind) 
                    prev_conf = conf.pop(ind)
                    component.insert(ind, merge_contour)
                    conf.insert(ind, max(prev_conf, solidity))
                    
                    #print("Prev conf = %f, current conf = %f, final conf = %f\n"%(prev_conf, solidity, max(prev_conf, solidity)))
            else:  
                if shape =='line' : 
                    #print("Detected a %s\n"%(shape))
                    cv2.drawContours(imcpy, [c], 0, (255, 255, 0), 2)
                    component.append(c)
                    conf.append(solidity)
                    #print("area = %d, hullarea = %d, solidity = %f" %(area, hullarea, solidity))      
                        
                elif shape =='rectangle' or shape =='square': 
                    #print("Detected a %s\n"%(shape))
                    cv2.drawContours(imcpy, [c], 0, (0, 255, 255), 2)
                    component.append(c)
                    conf.append(solidity)
                    #print("area = %d, hullarea = %d, solidity = %f" %(area, hullarea, solidity))      
                        
                elif shape =='pentagon' or shape =='hexagon': 
                    #print("................Detected a %s................\n"%(shape))
                    cv2.drawContours(imcpy, [c], 0, (0, 0, 255), 2)
                    component.append(c)
                    conf.append(solidity)
                    #print("area = %d, hullarea = %d, solidity = %f" %(area, hullarea, solidity))      
                        
                elif shape =='circle':# or shape =='ellipse': 
                    #print("Detected a %s\n"%(shape))
                    cv2.drawContours(imcpy, [c], 0, (0, 255, 0), 2)
                    component.append(c)
                    conf.append(solidity)
                    #print("area = %d, hullarea = %d, solidity = %f" %(area, hullarea, solidity))      
                        
                elif shape =='triangle' : 
                    #print("Detected a %s\n"%(shape))
                    cv2.drawContours(imcpy, [c], 0, (255, 0, 255), 2)
                    component.append(c)
                    conf.append(solidity)
                    #print("area = %d, hullarea = %d, solidity = %f" %(area, hullarea, solidity))      
                        
                      
        #                hull = cv2.convexHull(c, returnPoints = False)
        #                defects = cv2.convexityDefects(c, hull)
        #                print(defects)
        #    cv2.imshow("shape_detect", imcpy)
        #    cv2.waitKey(0)
        return (component, conf)
        
    def local_non_max_supp(self, contour, overlapThresh, conf):
        if len(contour) == 0:
            return []
            
        boxes = self.convert_contour2box(contour)
        
        
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
              
        areas = (x2-x1)*(y2-y1)
        indexes = np.argsort(conf) # np.argsort(y2)
        boxes_keep_index = []
                
        while len(indexes) > 0:
            last = len(indexes) - 1
            i = indexes[last]            
            boxes_keep_index.append(i)
            
            # find the largest (x, y) coordinates for the start of the bounding box 
            # and the smallest (x, y) coordinates for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[indexes[:last]])
            yy1 = np.maximum(y1[i], y1[indexes[:last]])
            xx2 = np.minimum(x2[i], x2[indexes[:last]])
            yy2 = np.minimum(y2[i], y2[indexes[:last]])
            
            w = np.maximum(xx2 - xx1, 0)
            h = np.maximum(yy2 - yy1, 0)
            intersections =  w*h
            unions = areas[indexes[:last]]
            overlap= intersections / unions
            indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
        keep_contour = [contour[p] for p in boxes_keep_index]
        
        return keep_contour
        
    def whichShape(self,c,img):
        shape = 'unknown'
        # calculate perimeter using
        peri = cv2.arcLength(c, True)
        # apply contour approximation and store the result in vertices
        vertices = cv2.approxPolyDP(c, 0.02* peri, True) # 0.015
#        cv2.drawContours(img, [vertices], 0, (255, 0, 255), 4)
#        cv2.imshow("contour", img)
#        cv2.waitKey(0) 
        
        if len(vertices) == 2: 
            shape = 'line'
    
        elif len(vertices) == 3:
            shape = 'triangle'
    
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(vertices) == 4:
            x, y, width, height = cv2.boundingRect(vertices)
            aspectRatio = float(width) / height
    
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                shape = "square"
            else:
                shape = "rectangle"
    
        # if the shape is a pentagon, it will have 5 vertices
        elif len(vertices) == 5:
            shape = "pentagon"
        
        elif len(vertices) == 6:
            shape = "hexagon"
    
        elif 7<len(vertices) < 15:
            shape = "ellipse"
        # otherwise, we assume the shape is a circle
        elif len(vertices) > 30:
            shape = "circle"
            
        return shape

    def find_component(self, filename, op_dir, im, thresh_im, gray_imcv):
        
        imcpy = im.copy()
        thresh_imcpy = thresh_im.copy()
        gray_imcvcpy = gray_imcv.copy()
        
#        #run a 5x5 gaussian blur then a 3x3 gaussian blr
#        blur5 = cv2.GaussianBlur(imcpy,(3,3),0)
#        blur3 = cv2.GaussianBlur(imcpy,(1,1),0)
#        DoGim = blur5 - blur3
#    
#        cv2.imshow("DoGim", DoGim)
#        cv2.waitKey(0)
        
        (cnts, _) = cv2.findContours(thresh_imcpy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_component = []
        final_conf = []
        line_component = []
        line_conf = []
        watershed_component = []
        watershed_conf = []
        # loop over the contours, call whichShape() for each contour and write the name of shape in the center 
        for c in cnts:    	
            area = cv2.contourArea(c)        
            if self.min_component_area<area<self.max_component_area :                 
#                M = cv2.moments(c)
#                # Compute the centroid 
#                cX = int(M['m10'] / M['m00'])
#                cY = int(M['m01'] / M['m00'])                  
                shape = self.whichShape(c, im)
                (line_component, line_conf) = self.mergeOrAppend(c, shape, line_component, line_conf, imcpy, 0)  
                             
        #######################################################################################################
        #compute the exact Euclidean distance from every binary pixel to the nearest zero pixel, then find peaks in this distance map
        kernel = np.ones((5,5),np.uint8)
        thresh_imcpy = cv2.erode(thresh_imcpy,kernel,iterations = 1)
        D = ndimage.distance_transform_edt(thresh_imcpy)
#        cv2.imshow("D", D)
#        cv2.waitKey(0)
        localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh_imcpy)
        # perform a connected component analysis on the local peaks, using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh_imcpy)
        #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        # loop over the unique labels returned by the Watershed algorithm
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background' so simply ignore it
            if label == 0:
                continue     
            # otherwise, allocate memory for the label region and draw it on the mask
            mask = np.zeros(gray_imcvcpy.shape, dtype="uint8")
            mask[labels == label] = 255         
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)            
            if self.min_component_area<area<self.max_component_area :
                 shape = self.whichShape(c, im)
                 (watershed_component, watershed_conf) = self.mergeOrAppend(c, shape, watershed_component, watershed_conf, imcpy, 1)                        
               
        final_component = line_component +  watershed_component
        final_conf = line_conf +  watershed_conf
        nmcomponent = self.local_non_max_supp(final_component, self.overlapTh, final_conf)
#        im_nmcomponent = im.copy()
#        for c in nmcomponent:
#            cv2.drawContours(im_nmcomponent, [c], 0, (0, 0, 100), 4)
#        cv2.imshow("im_nmcomponent", im_nmcomponent)
#        cv2.waitKey(0) 
            
        op_file_name = os.path.join(op_dir, "op" + os.path.basename(filename))
        cv2.imwrite(op_file_name, imcpy)
        
        
#        cv2.imshow("shape_detect", imcpy)
#        cv2.waitKey(0) 
        return nmcomponent
from __future__ import print_function
import numpy as np
import imutils
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import os
from Rectangle import Rectangle
from RectangleMerger import RectangleMerger as Merger
from scipy.signal import find_peaks

# Detects components from diagram image

class ShapeDetect:
    
    def __init__(self):
        self.min_component_area = 700 # Components with lower size will not be considered for processing
        self.max_component_area = 25000 # Components with higher size will not be considered for processing
        self.overlapTh = 0.50 # For non-maxima supression
        

    def convertContour2Box(self, cnt):
        box = []
        for c in cnt:
            x,y,w,h = cv2.boundingRect(c)
            box.append([x, y, x+w, y+h])
        return np.array(box)
    
    def convertContour2Rect(self, cnt):
        rect = []
        for c in cnt:
            x,y,w,h = cv2.boundingRect(c)
            rect.append(Rectangle.from_2_pos(x, y, x+w, y+h))
        return np.array(rect)
        
    def getMergeIndex(self, component, c):
        merger = Merger()
        rects = self.convertContour2Rect(component)
        x, y, w, h = cv2.boundingRect(c)
        index = []
        for i in range(0, len(rects)):
            if merger._is_rectangles_overlapped_horizontally(rects[i], Rectangle.from_2_pos(x, y, x+w, y+h)):
                index.append(i)
                
        return index
        
    # Sort Contours on the basis of their x-axis coordinates in ascending order
    def sortContours(self, cnts, method="horizontal"):

        if len(cnts) > 0:
    
            # initialize the reverse flag and sort index
            reverse = False
            i = 0
            if method == "vertical":
                i = 1
            # construct the list of bounding boxes and sort them from top to bottom
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            # print(len(cnts))
            # print(len(boundingBoxes))
            (cnts_new, boundingBoxes_new) = zip(*sorted(zip(cnts, boundingBoxes),
                                                key=lambda b: b[1][i], reverse=reverse))
            # return the list of sorted contours
            return cnts_new
            
        else:
            return []


    def smooth(self, x,window_len=11,window='hanning'):
  
        if x.ndim != 1:
            raise(ValueError, "smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise(ValueError, "Input vector needs to be bigger than window size.")
  
        if window_len<3:
            return x
            
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
  
        s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.'+window+'(window_len)')
  
        y = np.convolve(w/w.sum(),s,mode='valid')
        return y    
    
    def mergeOrAppend(self, c, shape, component, conf, imcpy, merge = 0):
        hull = cv2.convexHull(c)                 
        x, y, width, height = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        hullarea = cv2.contourArea(hull)
        solidity = float(area)/float(hullarea)
        
        merger = Merger()  
        mergable_index = []
        if merge == 1:
            mergable_index = self.getMergeIndex(component, c)
        if solidity > 0.4:
            if len(mergable_index) != 0:
                
                for ind in mergable_index:                    
                    mergable_conponent = component[ind]
                    x1, y1, w1, h1 = cv2.boundingRect(c)
                    x2, y2, w2, h2 = cv2.boundingRect(mergable_conponent)
                    merge_rec = merger._merge_2_rectangles(Rectangle.from_2_pos(x1, y1, x1+w1, y1+h1), Rectangle.from_2_pos(x2, y2, x2+w2, y2+h2))
                    merge_contour = np.array([[[merge_rec.x1, merge_rec.y1]],[[merge_rec.x2, merge_rec.y1]],[[merge_rec.x2, merge_rec.y2]], [[merge_rec.x1, merge_rec.y2]]], dtype=np.int32)
                   
                    component.pop(ind) 
                    prev_conf = conf.pop(ind)
                    component.insert(ind, merge_contour)
                    conf.insert(ind, max(prev_conf, solidity))
                                        
            else:  
                if shape =='line' :                     
                    cv2.drawContours(imcpy, [c], 0, (255, 255, 0), 2)
                    component.append(c)
                    conf.append(solidity)                        
                        
                elif shape =='rectangle' or shape =='square':                     
                    cv2.drawContours(imcpy, [c], 0, (0, 255, 255), 2)
                    component.append(c)
                    conf.append(solidity)                        
                        
                elif shape =='pentagon' or shape =='hexagon':                     
                    cv2.drawContours(imcpy, [c], 0, (0, 0, 255), 2)
                    component.append(c)
                    conf.append(solidity)                         
                        
                elif shape =='circle':# or shape =='ellipse':                     
                    cv2.drawContours(imcpy, [c], 0, (0, 255, 0), 2)
                    component.append(c)
                    conf.append(solidity)
                                              
                elif shape =='triangle' :                     
                    cv2.drawContours(imcpy, [c], 0, (255, 0, 255), 2)
                    component.append(c)
                    conf.append(solidity)
                     
                     
        return (component, conf)
        
    def localNonMaxSupp(self, contour, overlapThresh, conf):
        if len(contour) == 0:
            return []
            
        boxes = self.convertContour2Box(contour)
        
        
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
              
        areas = (x2-x1)*(y2-y1)
        indexes = np.argsort(conf) 
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
        
    def findFlowDir(self, img):
         
        height, width = img.shape[:2]
        sumX = np.sum(255-cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1) 
        sumX = self.smooth(sumX, window_len = 15, window = 'hanning')
        peakX, props = find_peaks(sumX, distance = 20, prominence = 2000)# threshold = 5000)
        cntPeakX = len(peakX)
        #calculate horizontal projection
        sumY = np.sum(255-cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0)
        sumY = self.smooth(sumY, window_len = 15, window = 'hanning')
        peakY, props = find_peaks(sumY, distance = 20, prominence = 2000)# threshold = 5000)
        cntPeakY = len(peakY)
        return "vertical" if cntPeakX>cntPeakY else "horizontal"
        
    def whichShape(self,c,img):
        shape = 'unknown'
        
        peri = cv2.arcLength(c, True)
        # apply contour approximation 
        vertices = cv2.approxPolyDP(c, 0.02* peri, True) # 0.015
        
        if len(vertices) == 2: 
            shape = 'line'
    
        elif len(vertices) == 3:
            shape = 'triangle'
    
        # if the shape has 4 vertices, it is either a square or a rectangle
        elif len(vertices) == 4:
            x, y, width, height = cv2.boundingRect(vertices)
            aspectRatio = float(width) / height
    
            # a square will have an aspect ratio that is approximately ~1, otherwise, itis a rectangle
            if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                shape = "square"
            else:
                shape = "rectangle"
    
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
        
        ################################################ Find components from line drawing ##########################################
        # (_, cnts, h) = cv2.findContours(thresh_imcpy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #
        (cnts, h) = cv2.findContours(thresh_imcpy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #
        final_component = []
        final_conf = []
        line_component = []
        line_conf = []
        watershed_component = []
        watershed_conf = []
        # loop over the contours, call whichShape() for each contour to find the shape
        for c in cnts:    	
            area = cv2.contourArea(c)        
            if self.min_component_area<area<self.max_component_area :               
                              
                shape = self.whichShape(c, im)
                (line_component, line_conf) = self.mergeOrAppend(c, shape, line_component, line_conf, imcpy, 0)  
                             
        ########################################### Find components from colored boxes #############################################
        #compute the exact Euclidean distance from every binary pixel to the nearest zero pixel, then find peaks in this distance map
        kernel = np.ones((5,5),np.uint8)
        thresh_imcpy = cv2.erode(thresh_imcpy,kernel,iterations = 1)
        D = ndimage.distance_transform_edt(thresh_imcpy)
        localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh_imcpy)
        # perform a connected component analysis on the local peaks, using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh_imcpy)
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
                 ################# add new components to the list only if it is not overlapping with existing ones ###################
                 (watershed_component, watershed_conf) = self.mergeOrAppend(c, shape, watershed_component, watershed_conf, imcpy, 1)                        
        

        ################################# Final components will be from line components and colored boxes #############################       
        final_component = line_component +  watershed_component
        final_conf = line_conf +  watershed_conf

        ############################### Apply non-max supression to get rid of redundant components ###################################
        nmcomponent = self.localNonMaxSupp(final_component, self.overlapTh, final_conf)
        flow_dir = self.findFlowDir(imcpy)
        sorted_contours = self.sortContours(nmcomponent, flow_dir)
#        im_nmcomponent = im.copy()
#        for c in nmcomponent:
#            cv2.drawContours(im_nmcomponent, [c], 0, (0, 0, 100), 4)
#        cv2.imshow("im_nmcomponent", im_nmcomponent)
#        cv2.waitKey(0) 
                    
#        cv2.imshow("shape_detect", imcpy)
#        cv2.waitKey(0) 
        return sorted_contours, flow_dir

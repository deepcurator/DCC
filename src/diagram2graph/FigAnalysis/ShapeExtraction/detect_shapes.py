from __future__ import print_function
import numpy as np
import imutils
import cv2
import argparse
from PIL import Image, ImageEnhance, ImageFilter
import re
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import glob
import os

def preprocessImage(image_path):

    # load the image on disk and then display it
    image = cv2.imread(image_path)
    imgPIL = Image.open(image_path)
    imgPIL = ImageEnhance.Color(imgPIL)
    imgPIL = imgPIL.enhance(0)
    gray_im = imgPIL.convert('L')  # convert image to monochrome
    # gray_im = ImageEnhance.Sharpness(gray_im).enhance(2)
    # gray_im = ImageEnhance.Contrast(gray_im).enhance(2)

    gray_imcv = np.array(gray_im, dtype=np.uint8)

    _, thresh_im = cv2.threshold(gray_imcv, 240, 255, cv2.THRESH_BINARY_INV)

    # Find edges in the image using canny edge detection method
    # Calculate lower threshold and upper threshold using sigma = 0.33
    # sigma = 0.33
    # v = np.median(thresh_im)
    # low = int(max(0, (1.0 - sigma) * v))
    # high = int(min(255, (1.0 + sigma) * v))
    # edged = cv2.Canny(thresh_im, low, high)
    
    return image, thresh_im, gray_imcv


def whichShape(c):

    shape = 'unknown'
    # calculate perimeter using
    peri = cv2.arcLength(c, True)
    # apply contour approximation and store the result in vertices
    vertices = cv2.approxPolyDP(c, 0.05 * peri, True)

    # If the shape it triangle, it will have 3 vertices
    if len(vertices) == 2:
        shape = 'line'

    elif len(vertices) == 3:
        shape = 'triangle'

    # if the shape has 4 vertices, it is either a square or a rectangle
    elif len(vertices) == 4:
        # using the boundingRect method calculate the width and height
        # of enclosing rectange and then calculte aspect ratio

        x, y, width, height = cv2.boundingRect(vertices)
        aspectRatio = float(width) / height

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"

    elif len(vertices) == 5:
        shape = "pentagon"
    
    elif len(vertices) == 6:
        shape = "hexagon"

    elif 6<len(vertices) < 15:
        shape = "Ellipse"
    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape

def detectShape(cnt, image):
    area = cv2.contourArea(cnt)
            
    if 700<area<20000:# and cv2.isContourConvex(c): 
        
        # compute the moment of contour
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # set values as what you need in the situation
            cX, cY = 0, 0
                    
        # call detectShape for contour c
        shape = whichShape(cnt)
            
        if shape == 'rectangle'or shape == 'square':
            # Outline the contours
            cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
        
            # Write the name of shape on the center of shapes
            cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
        
    return image


#####################################  Classify Images #############################################
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
        
        image, thresh_im, gray_imcv = preprocessImage(filename)
        (_, cnts, _) = cv2.findContours(thresh_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours, call detectShape() for each contour and write the name of shape in the center 
        for c in cnts:
        	image = detectShape(c, image)
                    
        # compute the exact Euclidean distance from every binary pixel to the nearest zero pixel, then find peaks in this distance map
        kernel = np.ones((5,5),np.uint8)
        thresh_im = cv2.erode(thresh_im,kernel,iterations = 1)
        D = ndimage.distance_transform_edt(thresh_im)
        localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh_im)         
        # perform a connected component analysis on the local peaks, using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh_im)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        # loop over the unique labels returned by the Watershed algorithm
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background' so simply ignore it
            if label == 0:
                continue         
            # otherwise, allocate memory for the label region and draw it on the mask
            mask = np.zeros(gray_imcv.shape, dtype="uint8")
            mask[labels == label] = 255         
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)            
            image = detectShape(c, image)
        

        op_file_name = os.path.join(outdir_path, "op" + os.path.basename(filename))
        print(op_file_name)
        cv2.imwrite(op_file_name, image)

        # show the output image
        #cv2.imshow("Output", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
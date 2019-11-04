from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import glob
import os

from ShapeDetect import ShapeDetect as sd
from ArrowDetect import ArrowDetect as ad
from TextDetect_OPENCV import TextDetectAll as tda
from Diag2Graph_v2 import Diag2Graph as tgv2
import pytesseract
from ParseJSON import ParseJSON as pj
from FigTypeDetect import FigTypeDetect as ftd
import subprocess
from subprocess import TimeoutExpired
# Comment below line for LINUX - Update below path for WINDOWS
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from image_rdf_generator import runI2G



# os.environ['http_proxy'] = "194.138.0.9:9400" 
# os.environ['https_proxy'] = "194.138.0.9:9400"


def preprocessImage(image_path, resize):
    
    # load the image from disk and then preprocess it
    image = cv2.imread(image_path)
    # add white border in the original image        
    image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[255, 255, 255]) 
    if resize == 1:
        newX, newY = image.shape[1]*1.5, image.shape[0]*1.5
    else:
        newX, newY = image.shape[1], image.shape[0]
            
    image_resize = cv2.resize(image,(int(newX),int(newY)))
        
    imgPIL = Image.open(image_path)
    imgPIL = ImageOps.expand(imgPIL, border = 10, fill = 'white')
    imgPIL = imgPIL.resize((int(newX),int(newY)), Image.ANTIALIAS)        
    imgPIL = ImageEnhance.Color(imgPIL)
    imgPIL = imgPIL.enhance(0)
    gray_im = imgPIL.convert('L') 

    gray_imcv = np.array(gray_im, dtype=np.uint8)    
    _, thresh_im = cv2.threshold(gray_imcv, 240, 255, cv2.THRESH_BINARY_INV)    
        
    return image_resize, thresh_im, gray_imcv


def run(input_path, op_path, ontology_file, model_dir):
    
    i2g_output_dir = op_path + "/image2graph"
    
    if not os.path.exists(i2g_output_dir):
        os.makedirs(i2g_output_dir)
        
    op_path_all = i2g_output_dir + "/all_images"
    
    if not os.path.exists(op_path_all):
        os.makedirs(op_path_all)

    
    # command = 'java -cp "pdffigures2_2.12-0.1.0.jar;pdffigures2-assembly-0.1.0-deps.jar;scala-library.jar" org.allenai.pdffigures2.FigureExtractorBatchCli Input/ -s stat_file.json -m out/ -d out/'
    command = 'java -cp "pdffigures2_2.12-0.1.0.jar;pdffigures2-assembly-0.1.0-deps.jar;scala-library.jar" org.allenai.pdffigures2.FigureExtractorBatchCli ' + input_path + '/ -s stat_file.json -m ' + op_path_all + '/ -d ' + op_path_all + '\\'
    # , stderr=subprocess.PIPE
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
#    print(stderr)
    
    
    print("[INFO] Loading trained models ...")
            
    figtypedetector = ftd(model_dir)
    figtypedetector.loadFigClassModels("vgg16")
    
    
    #if (process.returncode == 0):
    print("[INFO] Loading and analyzing images ...")
    
#    input_path = "Input"
#    op_path = "out"
    

    
    
    for filename in glob.glob(os.path.join(op_path_all, '*png')):
#        image_file_name = os.path.splitext(os.path.basename(filename))[0]
#        abspath = os.path.abspath(filename)
        filename = filename.replace("\\", "/")
        # print("Processing image: " + filename)
        if (filename.find('Figure') != -1): 
            parsejson = pj()
            # paper_title, paper_file_name, paper_conf, paper_year, fig_caption, fig_text = parsejson.getCaption(filename)
            fig_caption, fig_text, paper_file_name = parsejson.getCaption_noCSV(filename)
            
            figTypeResult = parsejson.isResult(fig_caption)
            figTypeDiag = parsejson.isDiag(fig_caption)
            
#            print(paper_file_name)
#            print(figTypeResult)
#            print(figTypeDiag)
            
            if (not figTypeResult and figTypeDiag):
                im, thresh_im, gray_imcv = preprocessImage(filename, 0)
                binType, mcType = figtypedetector.detectFigType(im)
                    
                if mcType < 3:
                    # print(os.path.join(i2g_output_dir, paper_file_name))
                    if not os.path.isdir(os.path.join(i2g_output_dir, paper_file_name)):
                        os.mkdir(os.path.join(i2g_output_dir, paper_file_name))
                        os.mkdir(os.path.join(i2g_output_dir, paper_file_name, "diag2graph"))
                        os.mkdir(os.path.join(i2g_output_dir, paper_file_name, "Figures"))
    
                    cv2.imwrite(os.path.join(i2g_output_dir, paper_file_name+ "/Figures/" + os.path.basename(filename)), im)        
    
    
                    shapedetector = sd()
                    component, flow_dir = shapedetector.find_component(filename, i2g_output_dir, im, thresh_im, gray_imcv)
    
                    textdetector = tda()
                    text_list = textdetector.combinedTextDetect(filename, im, component, fig_text)
    
                    arrowdetector = ad()            
                    line_connect = arrowdetector.detectLines(im, thresh_im, gray_imcv, component, text_list)
    
                    graphcreator = tgv2()
                    # createDiag2Graph(self, op_dir, filename, img, thresh_im, comp, flow_dir, text_list, line_list, paper_title, paper_file_name, paper_conf, paper_year, fig_caption)
                    graphcreator.createDiag2Graph(i2g_output_dir, filename, im, thresh_im, component, flow_dir, text_list, line_connect, None, paper_file_name, None, None, fig_caption)
    
    #else:
    
        #print("Pdf2Fig Terminated with Status %d. Exiting."% (process.returncode)   )
    
    # image_triple_dir = "C:\\dcc_test\\src\\diagram2graph\\FigAnalysis\\ShapeExtraction\\Output\\"
    # image_output_dir = "C:\\dcc_test\\src\\diagram2graph\\FigAnalysis\\ShapeExtraction\\Output\\"
    
    print("[INFO] Creating RDF graph ...")
    
    # paper_dir = "C:/dcc_test/src/diagram2graph/FigAnalysis/ShapeExtraction/Input/"
    # image_triple_dir = "C:/dcc_test/src/diagram2graph/FigAnalysis/ShapeExtraction/out/"
    # image_output_dir = "C:/dcc_test/src/diagram2graph/FigAnalysis/ShapeExtraction/out/"
    
    runI2G(input_path, i2g_output_dir, op_path_all, ontology_file)
    
    print("[Info] Completed image2graph pipeline!")

#inputFolder = 'demo_input'
#outputFolder = 'Output_a'
## outputFolderImages = 'Output'
##
#run(inputFolder, outputFolder)
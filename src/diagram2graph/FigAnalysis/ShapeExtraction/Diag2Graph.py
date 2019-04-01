import cv2
import os
import numpy as np
from ArrowDetect import ArrowDetect as ad
import re

class Diag2Graph:
    
    def __init__(self):
        self.min_dist_init = 1000000
        self.min_dist_thresh = 75 # 0.1
        self.layer_type = ["input", "dense", "conv", "conv2D", "conv 2D", "flatten", "dropout", "max pool", "avg pool", "maxpool", "avgpool", "max pooling", "avg pooling", "maxpooling", "avgpooling", "concat", "embed", "rnn", "LSTM", "output"]
        
    def find_closest_component_rect(self, rectx, recty, cnt):
        
        min_dist = self.min_dist_init 
        nearest_comp = None
        for index in range(len(cnt)):
            c = cnt[index]
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            dx = cX-rectx/2
            dy = cY-recty/2
            D = np.sqrt(dx*dx + dy*dy)
            if D < min_dist:
                min_dist = D
                nearest_comp = index
        return (nearest_comp, min_dist)
        
    
        
    def getTextCompTag(self, comp, text_list):
        compWithText = {}  
        TextWOComp = []
        compWOText = [] 
        
        for ((startX, startY, endX, endY), op, prob) in text_list:
            neighbor_comp_index, neighbor_comp_dist = self.find_closest_component_rect(startX+endX, startY+endY, comp)
            if neighbor_comp_dist < self.min_dist_thresh: 
                if neighbor_comp_index in compWithText.keys():
                    compWithText[neighbor_comp_index].append(op)
                else:
                    compWithText[neighbor_comp_index] = [op]
            else:
                TextWOComp.append(((startX, startY, endX, endY), op, prob))    
                
        for index in range(len(comp)):
            if index not in compWithText.keys():
                compWOText.append(index)
          
        return (compWithText, TextWOComp, compWOText)
    
    def find_layerName(self, txt) :
        txt = txt.lower()
        
        for layer in self.layer_type:
           
            if (re.search(layer, txt, flags = 0) is not None )or (re.search(txt, layer, flags = 0) is not None):
                return (True, layer, re.sub(layer, "", txt))
        return (False, "", txt)

    
    def createDiag2Graph(self, op_dir, filename, img, thresh_im, comp, text_list, line_list):
        
        op_image_name = os.path.join(op_dir, "OpImage/op" + os.path.basename(filename))
        op_file_name = os.path.join(op_dir, "OpGraph/graph" + os.path.splitext(os.path.basename(filename))[0] + ".txt")
        op_file = open(op_file_name, "w")
        
        final_graph_im = img.copy()
        (compWithText, TextWOComp, compWOText) =  self.getTextCompTag(comp, text_list)  
         
        arrowdetector = ad()            
        line_connect = arrowdetector.getLineCompTag(img, thresh_im, comp, line_list, TextWOComp)
        
        for k in compWithText.keys():
            cnt = comp[k]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
                
            cv2.putText(final_graph_im, str(k), (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)             
            cv2.drawContours(final_graph_im, [cnt], 0, (255, 0, 255), 2)            
            print("=======+++++++++++++++++++++=======")
            text_pos = 0
            for txt in compWithText[k]:
                found, layer_name, remaining_txt = self.find_layerName(txt)
                if found:
                    print("Component number %d \"has type\": %s \n"% (k, layer_name))
                    op_file.write("Component number %d \"has type\": %s \n"% (k, layer_name))
                    print("Component number %d \"has description\": %s \n"% (k, remaining_txt))    
                    op_file.write("Component number %d \"has description\": %s \n"% (k, remaining_txt))
                
                else:
                    print("Component number %d \"has description\": %s \n"% (k, txt))    
                    op_file.write("Component number %d \"has description\": %s \n"% (k, txt))
                
                cv2.putText(final_graph_im, txt, (cX, cY + text_pos), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 255), 2)
                # cv2.imshow("final_graph_im", final_graph_im)
                # cv2.waitKey(0)
                text_pos +=15
                
        for index in compWOText:  #for k in results_wotext:
            print("----------------Component number %d has text: None --------------------"%(index));
            op_file.write("Component number %d has text: None \n"% (index))
            
            cnt = comp[index]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
                
            cv2.drawContours(final_graph_im, [cnt], 0, (0, 0, 255), 2) 
            cv2.putText(final_graph_im, str(index), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)             
            # cv2.imshow("final_graph_im", final_graph_im)
            # cv2.waitKey(0)
    
        for ((startX, startY, endX, endY), op, prob) in TextWOComp:
            print("................ Text without Component: %s....................."%(op));
            op_file.write("Text Component: %s \n"% (op))
                 
            cv2.rectangle(final_graph_im, (startX, startY), (endX, endY), (0, 200, 0), 2)
            cv2.putText(final_graph_im, op, (startX+10, startY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imshow("final_graph_im", final_graph_im)
            # cv2.waitKey(0)
             
        
        
         
        for ((startx, starty, endx, endy), start_comp_found, end_comp_found, start_text_found, end_text_found) in line_connect:
            if (start_comp_found != -1 and end_comp_found != -1):
                print("Component number %d \"connected to\" Component number %d \n"% (start_comp_found, end_comp_found)) 
                op_file.write("Component number %d \"connected to\" Component number %d \n"% (start_comp_found, end_comp_found)) 
            elif(start_text_found != -1 and end_text_found != -1):
                print("Text Box %s \"connected to\" TextBox %s \n"% (TextWOComp[start_text_found][1], TextWOComp[end_text_found][1])) 
                op_file.write("Text %s \"connected to\" Text %s \n"% (TextWOComp[start_text_found][1], TextWOComp[end_text_found][1])) 
            elif(start_comp_found != -1 and end_text_found != -1):
                print("Component Number %d \"connected to\" TextBox %s \n"% (start_comp_found, TextWOComp[end_text_found][1])) 
                op_file.write("Component Number %d \"connected to\" Text %s \n"% (start_comp_found, TextWOComp[end_text_found][1])) 
            elif(end_comp_found != -1 and start_text_found != -1):
                print("Component Number %d \"connected to\" TextBox %s \n"% (end_comp_found, TextWOComp[start_text_found][1])) 
                op_file.write("Component Number %d \"connected to\" Text %s \n"% (end_comp_found, TextWOComp[start_text_found][1])) 

            cv2.line(final_graph_im, (startx, starty), (endx,endy), (0,200,255), 2) 
            # cv2.imshow("final_graph_im", final_graph_im)
            # cv2.waitKey(0)
        
        cv2.imwrite(op_image_name, final_graph_im)
        op_file.close()       
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%");
                    
        #cv2.imshow("final_graph_im", final_graph_im)
        #cv2.waitKey(0)   
            
        

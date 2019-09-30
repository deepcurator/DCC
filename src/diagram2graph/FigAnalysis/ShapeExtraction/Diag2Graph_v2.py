# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from ArrowDetect import ArrowDetect as ad
import re
import difflib

from Rectangle import Rectangle
from RectangleMerger import RectangleMerger as Merger

class Diag2Graph:
    
    def __init__(self):
        self.min_dist_init = 1000000
        self.min_dist_thresh = 15
        self.layer_type = ["input", "dense", "conv", "conv2D", "conv 2D", "flatten", "dropout", "max pool", "avg pool", "maxpool", "avgpool", "max pooling", 
                           "avg pooling", "maxpooling", "avgpooling", "concat", "embed", "rnn", "LSTM", "rnnseq", "LSTMseq", "output"]
        self.layer_type_stardard = {"input":"input",
                                    "conv2D":"conv", "conv-2D":"conv", "conv 2D":"conv", "convolutional":"conv", "convolution":"conv", "conv":"conv",
                                    "deconv2D":"deconv", "deconv-2D":"deconv", "deconv 2D":"deconv", "deconvolutional":"deconv", "deconvolution":"deconv", "transpose convolution": "deconv", "transpose conv": "deconv", "deconv": "deconv",
                                    "normalizer":"norm", "l1norm":"norm", "l2norm":"norm", "normalize":"norm", "normalization":"norm", "batchnormalize":"norm", "batchnormalization":"norm", "batch normalize": "norm", "batch normalization":"norm", "batchnorm":"norm", "batch norm":"norm", "batch-normalize": "norm", "batch-normalization":"norm", "batch-norm":"norm", "norm":"norm", 
                                    "leakyRelu":"activation", "leaky relu":"activation", "leaky-Relu":"activation", "relu":"activation", "sigmoid":"activation", "tanh":"activation", "softmax":"activation", "activation":"activation", 
                                    "drop out":"dropout", "dropout":"dropout",
                                    "fc6": "dense","fc7":"dense","fc8":"dense", "fc9":"dense", "fullyconnected": "dense", "fully connected": "dense", "fully-connected": "dense", "fullconnected": "dense", "full connected": "dense", "full-connected": "dense", "fullconnection": "dense", "full connection": "dense", "full-connection": "dense", "dense": "dense", 
                                    "max pooling": "pooling", "max-pooling": "pooling", "maxpooling 2d": "pooling", "maxpooling2d": "pooling",  "maxpool2d": "pooling", "maxpool 2d": "pooling",  "max pool": "pooling", "max-pool": "pooling", 
                                    "avg pooling": "pooling", "avg-pooling": "pooling", "avgPool 2d": "pooling", "avgpooling 2d": "pooling","avgpooling2d": "pooling", "avgpooling": "pooling", "avgpool2d": "pooling", "avgpool": "pooling",  "avg-pool": "pooling", "pooling": "pooling",
                                    "upsample":"unpooling", "upsampleing":"upsample", "unpooling":"unpooling",
                                    "logsoftmax": "loss", "softmaxwithloss": "loss","sigmoid": "loss", "softmax": "loss", "loss": "loss",
                                    "flatten": "flatten", 
                                    "concatenation":"concat", "concat":"concat", 
                                    "rnn": "rnn", "recurrent network": "rnn",  "recurrent-network": "rnn",  
                                    "lstm": "lstm",
                                    "recurrent sequence": "rnnseq", "recurrent-sequence": "rnnseq", "rnn sequence": "rnnseq", "rnn-sequence": "rnnseq", "rnnsequence": "rnnseq",  "rnn-seq": "rnnseq", "rnnseq": "rnnseq", 
                                    "lstm-seq": "lstmseq", "lstmsequence": "lstmseq", "lstm sequence": "lstmseq", "lstm-sequence": "lstmseq", "lstmseq": "lstmseq", 
                                    "output": "output"}
                                    
        
        self.valid_conn = {"input":("dense", "conv", "embed", "rnn", "lstm", "norm", "concat"), 
                           "conv": ("conv", "flatten", "dropout", "pooling", "concat", "norm", "activation"),
                           "deconv": ("deconv", "unpooling", "norm", "activation"),
                           "dense": ("dense", "dropout", "concat", "embed", "loss","norm", "activation","output"),  
                           "flatten": ("dense","dropout", "concat", "embed", "norm"), 
                           "dropout": ("dense","dropout", "concat", "embed", "lstm", "rnn", "lstmseq", "rnnseq"), 
                           "pooling": ("conv", "flatten", "dropout", "pooling", "concat", "dense"),
                           "unpooling": ("conv", "deconv"),
                           "concat": ("dense", "dropout", "concat", "embed", "conv", "deconv", "flatten", "pooling", "rnn", "rnnseq", "LSTM", "LSTMseq", "norm"),                            
                           "rnn": ("dense", "dropout", "concat", "embed", "loss","norm", "output"), 
                           "rnnseq": ("flatten", "dropout", "concat", "rnn", "rnnseq", "lstm", "lstmseq", "output"),
                           "lstm": ("conv", "dropout", "concat", "embed", "loss","norm", "output"), 
                           "lstmseq": ("flatten", "dropout", "concat", "rnn", "rnnseq", "lstm", "lstmseq", "output"),
                           "norm": ("dense", "relu"),
                           "embed": ("concat", "rnn", "lstm", "rnnseq", "lstmseq"),
                           "activation": ("flatten", "dropout", "upsample", "conv", "pooling", "loss"),
                           "loss": ("output")}

   
    def find_min_dist_2rec(self, start_BB, end_BB):
        sx1 = start_BB[0]
        sy1 = start_BB[1]
        ex1 = start_BB[2] + sx1 
        ey1 = start_BB[3] + sy1

        sx2 = end_BB[0]
        sy2 = end_BB[1]
        ex2 = end_BB[2] + sx2
        ey2 = end_BB[3] + sy2

        min_horizontal_dist = min( abs(sx1-sx2), abs(ex1-sx2), abs(sx1-ex2), abs(ex1-ex2) )
        min_vertical_dist = min( abs(sy1-sy2), abs(ey1-sy2), abs(sy1-ey2), abs(ey1-ey2) )

        return min(min_horizontal_dist, min_vertical_dist)


    def find_closest_component_rect(self, startX, startY, endX, endY, cnt):
        rectx = (startX+endX)/2
        recty = (startY+endY)/2

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
            dx = cX-rectx
            dy = cY-recty
            D = np.sqrt(dx*dx + dy*dy)
            if D < min_dist:
                min_dist = D
                nearest_comp = index

        if len(cnt) > 0:
            min_dist_pos = self.find_min_dist_2rec([startX, startY, endX-startX, endY-startY], cv2.boundingRect(cnt[nearest_comp]) )
            return (nearest_comp, min(min_dist, min_dist_pos))
        else:
            return (nearest_comp, min_dist)

    def isPresent(self, word, wordlist):

        for items in wordlist:
            if re.match('^[0-9a-zA-Z]*$', word.lower().replace(" ", ""))  and re.search(word.lower().replace(" ", ""), items.lower().replace(" ", ""), flags = 0) is not None:
                return 1
            elif re.match('^[0-9a-zA-Z]*$',items.lower().replace(" ", ""))  and re.search(items.lower().replace(" ", ""), word.lower().replace(" ", ""), flags = 0) is not None:
                return items 

        return 0

    def remove_duplicate(self, duplicate): 
        final_list = [] 

        for word in duplicate: 
            if self.isPresent(word, final_list) == 0:
                final_list.append(word) 
            elif self.isPresent(word, final_list)!= 0 and self.isPresent(word, final_list)!= 1:
                final_list.remove(self.isPresent(word, final_list))
                final_list.append(word)
            
        return final_list 
    
    def OverlapTextBBox(self, ipTextComp, ip_list):

        merge = Merger()


        ((SX, SY, EX, EY), ipop, ipprob, iptextID) = ipTextComp
        final_list = ip_list.copy()

        for ((startX, startY, endX, endY), op, prob, textID) in ip_list:
            if merge._is_rectangles_overlapped(Rectangle.from_2_pos(startX, startY, endX, endY), Rectangle.from_2_pos(SX, SY, EX, EY)):
                    rec = merge._merge_2_rectangles(Rectangle.from_2_pos(startX, startY, endX, endY), Rectangle.from_2_pos(startX, startY, endX, endY))
                    
                    mergeOp = " ".join(ipop.split() + list(set(op.split()) - set(ipop.split())))
                    mergeProb = (ipprob+prob)/2
                    mergeID = textID
                    final_list.remove(((startX, startY, endX, endY), op, prob, textID))
                    final_list.append(((rec.x1, rec.y1, rec.x2, rec.y2), mergeOp, mergeProb, mergeID))

                    return True, final_list

        return False, None  

    def remove_textDuplicate(self, TextWOComp):
        final_list = []
        for ((startX, startY, endX, endY), op, prob, textID) in TextWOComp:
            flag, updatedList = self.OverlapTextBBox(((startX, startY, endX, endY), op, prob, textID), final_list)
            if flag:
                final_list = updatedList
            else:
                final_list.append(((startX, startY, endX, endY), op, prob, textID)) 

        return final_list


    def getTextCompTag(self, comp, text_list, line_list):
        compWithText = {}  
        TextWOComp = []
        compWOText = [] 
        
        text_list= sorted(text_list, key=lambda r:r[0][1])

        textID = 0
        for ((startX, startY, endX, endY), op, prob) in text_list:
            neighbor_comp_index, neighbor_comp_dist = self.find_closest_component_rect(startX, startY, endX, endY, comp)
            if neighbor_comp_dist < self.min_dist_thresh: #*(max(np.size(img, 0), np.size(img, 1))):
                if neighbor_comp_index in compWithText.keys():
                    compWithText[neighbor_comp_index].append(op)
                else:
                    compWithText[neighbor_comp_index] = [op]
            else:
                TextWOComp.append(((startX, startY, endX, endY), op, prob, textID)) 
                textID+=1 
                
                
        for index in range(len(comp)):
            if index not in compWithText.keys():
                compWOText.append(index)

        for index in compWithText.keys():
            compWithText[index]= self.remove_duplicate(compWithText[index])
             
        return (compWithText, TextWOComp, compWOText)

    def find_dominant_flow(self, followedby_comps, node_list, flow_dir):
        updated_flow_dir = ""

        flow_cnt_forward = 0
        if flow_dir == "horizontal":
            #initialize flow as left-to-right
            updated_flow_dir = "left-to-right"
        else:
            #initialize flow as top-to-bottom for vertical flow
            updated_flow_dir = "top-to-bottom"


        for index in range(len(followedby_comps)):
            (start, end, flow) = followedby_comps[index]
            start_node = self.nodeSearch(start, node_list)
            end_node = self.nodeSearch(end, node_list)
            [sx, sy] = start_node['center']
            [ex, ey] = end_node['center']

            if flow_dir == "horizontal":
                if sx < ex: # left-to-right
                    flow_cnt_forward += 1
                else: # right-to-left
                    flow_cnt_forward -= 1
            else: # flow_dir == "vertical":
                if sy < ey: #top-to-bottom
                    flow_cnt_forward += 1
                else: #bottom-to-top
                    flow_cnt_forward -= 1

            if flow_cnt_forward < 0:
                if flow_dir == "horizontal":
                    #update flow as right-to-left
                    updated_flow_dir = "right-to-left"
                else:
                    #update flow as bottom-to-top for vertical flow
                    updated_flow_dir = "bottom-to-top"

        return updated_flow_dir


    def find_layerName(self, txt) :
        txt = txt.strip()
        txt = txt.lower()
        txt = re.sub("layer", "", txt).strip()
        
        for layer in self.layer_type_stardard:
           
            matcher = difflib.SequenceMatcher(None, layer, txt)
            match = matcher.find_longest_match(0, len(layer), 0, len(txt))
            if (match.size > 2 and match.size > 0.9*(min(len(layer), len(txt)))): 
                return (True, self.layer_type_stardard[layer], re.sub(layer, "", txt).strip())
        return (False, "", txt)
    
    def find_valid_layer_seq(self, comp1, comp2, layer_name1, layer_name2) :
        
        if layer_name1 and layer_name2:
            if layer_name1 in self.valid_conn:
               following_layer = self.valid_conn[layer_name1]
               if (layer_name2 in following_layer) : 
                   return (True, comp1, comp2)
            if layer_name2 in self.valid_conn:
               following_layer = self.valid_conn[layer_name2]
               if (layer_name1 in following_layer): 
                   return (True, comp2, comp1)
        
        return (False, "", "")
    
    def update_connect_info(self, connected_comps, node_list, flow_dir):

        for index in range(len(connected_comps)):
            (nodeID1, nodeID2, flow) = connected_comps[index]

            node1 = self.nodeSearch(nodeID1, node_list)
            node2 = self.nodeSearch(nodeID2, node_list)
            [x1, y1] = node1['center']
            [x2, y2] = node2['center']

            if flow_dir == "left-to-right":
                if x1<x2:
                    start_node = node1
                    end_node = node2
                else:
                    start_node = node2
                    end_node = node1

            elif flow_dir == "right-to-left":
                if x1<x2:
                    start_node = node2
                    end_node = node1
                else:
                    start_node = node1
                    end_node = node2
            elif flow_dir == "top-to-bottom":
                if y1<y2:
                    start_node = node1
                    end_node = node2
                else:
                    start_node = node2
                    end_node = node1
            elif flow_dir == "bottom-to-top":
                if y1<y2:
                    start_node = node2
                    end_node = node1
                else:
                    start_node = node1
                    end_node = node2

            start_node['next'] = end_node['nodeId']
            end_node['prev'] = start_node['nodeId']

        return node_list


    def nodeSearch(self, NID, node_list):
        #return [element for element in node_list if element['nodeId'] == NID]
        for node in node_list:
            if node['nodeId'] == NID:
                return node
                
        
    def nodeSequence(self, start_node, end_node, node_list, start_node_type = "Comp", end_node_type = "Comp" ):
        
        node1 = self.nodeSearch(start_node_type + str(start_node), node_list)
        node2 = self.nodeSearch(end_node_type + str(end_node), node_list)
                    
        if node1 and node2:
            connected, prev_layer, next_layer = self.find_valid_layer_seq(start_node, end_node, node1['layerName'], node2['layerName'])
            if connected:
                if prev_layer == int(start_node):
                    node1['next'] = end_node_type + str(end_node)
                    node2['prev'] = start_node_type + str(start_node)
                elif next_layer == int(start_node):
                    node1['prev'] = end_node_type + str(end_node)
                    node2['next'] = start_node_type + str(start_node)
                    
                return("Comp"+str(prev_layer), "Comp"+str(next_layer), "followedBy")                    
            else:
                return(start_node_type + str(start_node), end_node_type + str(end_node), "connectedTo")
                            
        else:
            return None, None, None


    # Sort Contours on the basis of their x-axis coordinates in ascending order
    def sort_contours(self, cnts, method="horizontal"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        if method == "vertical":
            i = 1
        # construct the list of bounding boxes and sort them from top to bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours
        return cnts

    def createDiag2Graph(self, op_dir, filename, img, thresh_im, comp, flow_dir, text_list, line_list, paper_title, paper_file_name, paper_conf, paper_year, fig_caption):
        
        print(filename)
        op_image_name = os.path.join(op_dir, paper_file_name + "/diag2graph/"+ os.path.basename(filename))
        op_file_name = os.path.join(op_dir, paper_file_name + "/diag2graph/" + os.path.splitext(os.path.basename(filename))[0] + '.txt')
        

        op_file = open(op_file_name, "w")
        FigureID = os.path.splitext(os.path.basename(filename))[0] #+"_"+paper_conf+"_"+paper_year
        
        op_file.write(":%s isA Figure \n"% (FigureID))
        op_file.write(":%s foundIn %s \n"% (FigureID, paper_title))
        op_file.write(":%s hasCaption %s \n"% (FigureID, fig_caption))
        final_graph_im = img.copy()
        (compWithText, TextWOComp, compWOText) =  self.getTextCompTag(comp, text_list, line_list)  
        arrowdetector = ad()            
        line_connect = arrowdetector.getLineCompTag(img, thresh_im, comp, line_list, TextWOComp)
        
        node_list = [] 
        #keys = ["nodeId", "layerName", "description", "location", "before", "after"]
        for k in compWithText.keys():
            temp_node = {}
            temp_node['nodeId'] = 'Comp' + str(k)

            op_file.write(":Comp%d partOf :%s \n"% (k, FigureID))

            cnt = comp[k]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            temp_node['center'] = [cX, cY]    
            temp_node['layerName'] = ""    
            temp_node['description'] = []
            cv2.putText(final_graph_im, str(k), (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)             
            cv2.drawContours(final_graph_im, [cnt], 0, (255, 0, 255), 2)            
            text_pos = 0

            op_file.write(":Comp%d hasPos %s \n"% (k, str(cv2.boundingRect(comp[k]))))

            for txt in compWithText[k]:
                if not temp_node['layerName']:
                    found, layer_name, remaining_txt = self.find_layerName(txt)
                    
                    if found:
                        op_file.write(":Comp%d isType %s \n"% (k, layer_name))
                        temp_node['layerName'] = layer_name 
                        temp_node['description'].append(txt) 
                    else:
                        temp_node['description'].append(txt) 
                elif (re.search("layer", txt, flags = 0) is None):
                    temp_node['description'].append(txt)               
                                      
                cv2.putText(final_graph_im, txt, (cX, cY + text_pos), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 255), 2)
                text_pos +=15
            op_file.write(":Comp%d hasDescription %s \n"% (k, temp_node['description']))    
            node_list.append(temp_node)
           
        regex = re.compile('@_!#$%^&/\~:')
        TextWOComp= self.remove_textDuplicate(TextWOComp)
        for ((startX, startY, endX, endY), op, prob, textID) in TextWOComp:
            if(len(op) > 1 and regex.search(op) == None):
                temp_node = {}
                temp_node['nodeId'] = 'Text' + str(textID)
                temp_node['center'] = [(startX+endX)/2, (startY+endY)/2] 
                temp_node['layerName'] = ""    
                temp_node['description'] = op
                node_list.append(temp_node)
                
                op_file.write(":Text%d partOf :%s \n"% (textID, FigureID))
                op_file.write(":Text%d hasPos %s \n"% (textID, str((startX, startY, endX-startX, endY-startY))))
                op_file.write(":Text%d hasDescription %s \n"% (textID, op.split()))

                cv2.rectangle(final_graph_im, (startX, startY), (endX, endY), (0, 200, 0), 2)
                cv2.putText(final_graph_im, op, (startX+10, startY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
             
        
        
        if line_connect:
            followedby_comps = []
            connected_comps = []
            
            for ((startx, starty, endx, endy), start_comp_found, end_comp_found, start_text_found, end_text_found) in line_connect:
                print("Diag2Graph:  start_comp_found = %s, end_comp_found = %s, start_text_found = %s, end_text_found = %s"%(start_comp_found, end_comp_found, start_text_found, end_text_found))

                if (start_comp_found != -1 and end_comp_found != -1):
                    s, e, conn = self.nodeSequence(start_comp_found, end_comp_found, node_list, start_node_type = "Comp", end_node_type = "Comp") 
                    if conn == "followedBy":
                        followedby_comps.append((s, e, conn)) 
                    elif conn == "connectedTo":
                        connected_comps.append((s, e, conn)) 
                elif (start_text_found != -1 and end_text_found != -1):
                    s, e, conn = self.nodeSequence(start_text_found, end_text_found, node_list, start_node_type = "Text", end_node_type = "Text")
                    if conn == "followedBy":
                        followedby_comps.append((s, e, conn)) 
                    elif conn == "connectedTo":
                        connected_comps.append((s, e, conn))
                elif(start_comp_found != -1 and end_text_found != -1):
                    s, e, conn = self.nodeSequence(start_comp_found, end_text_found, node_list, start_node_type = "Comp", end_node_type = "Text")
                    if conn == "followedBy":
                        followedby_comps.append((s, e, conn)) 
                    elif conn == "connectedTo":
                        connected_comps.append((s, e, conn))
                elif(end_comp_found != -1 and start_text_found != -1):
                    s, e, conn = self.nodeSequence(start_text_found, end_text_found, node_list, start_node_type = "Text", end_node_type = "Comp")
                    if conn == "followedBy":
                        followedby_comps.append((s, e, conn)) 
                    elif conn == "connectedTo":
                        connected_comps.append((s, e, conn))

                

            flow_dir_updated = self.find_dominant_flow(followedby_comps, node_list, flow_dir)

            node_list = self.update_connect_info(connected_comps, node_list, flow_dir_updated)

            op_file.write(":%s hasFlow %s \n"% (FigureID, flow_dir_updated))   
        
        
        cv2.imwrite(op_image_name, final_graph_im)
        op_file.close()       
            
        

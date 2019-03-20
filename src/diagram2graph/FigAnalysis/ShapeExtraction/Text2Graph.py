import cv2

import numpy as np
import re
import os
from rectangle import Rectangle
from rectangle_merger import RectangleMerger as Merger


class Text2Graph:
    
    def __init__(self):
        self.min_dist_init = 1000000
        self.min_dist_thresh = 75 # 0.1
        
    def find_closest_component(self, rectx, recty, cnt):
        
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
        #print("min dist = %d, index = %d \n"% (min_dist, nearest_comp))
        return (nearest_comp, min_dist)
                
        
    def createText2Graph(self, img, comp, text_list):
        
        final_results_withcomp = {}  
        final_results_wocomp = {}
        final_results_wotext = [] 
        final_graph_im = img.copy()
        for ((startX, startY, endX, endY), op, prob) in text_list:
            neighbor_comp_index, neighbor_comp_dist = self.find_closest_component(startX+endX, startY+endY, comp )
            #print("min_dist = %d, min_length = %d, thresh = %d"% (neighbor_comp_dist, max(np.size(img, 0), np.size(img, 1)), self.min_dist_thresh*max(np.size(img, 0), np.size(img, 1))))
            if neighbor_comp_dist < self.min_dist_thresh: #*(max(np.size(img, 0), np.size(img, 1))):
                #print(comp[c])
                if neighbor_comp_index in final_results_withcomp.keys():
                    final_results_withcomp[neighbor_comp_index].append(op)
                else:
                    final_results_withcomp[neighbor_comp_index] = [op]
            else:
                final_results_wocomp[(startX, startY, endX, endY)] = op
                
        for k in final_results_withcomp.keys():
            cnt = comp[k]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            cv2.drawContours(final_graph_im, [cnt], 0, (255, 0, 255), 2)
            
            print("=======+++++++++++++++++++++=======")
            text_no = 0
            for txt in final_results_withcomp[k]:
                print("Component number %d has text: %s \n"% (k, txt))    
                cv2.putText(final_graph_im, txt, (cX, cY + text_no), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 255), 2)
                cv2.imshow("final_graph_im", final_graph_im)
                cv2.waitKey(0)
                text_no +=15
                
        for k in final_results_wocomp.keys():
            print(k)
            print(final_results_wocomp[k])
            print(k[0]+10)
            cv2.rectangle(final_graph_im, (k[0], k[1]), (k[2], k[3]), (0, 0, 255), 1)
            cv2.putText(final_graph_im, final_results_wocomp[k], (k[0]+10, k[1]+10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 255), 2)
            cv2.imshow("final_graph_im", final_graph_im)
            cv2.waitKey(0)
        
        
        for index in range(len(comp)):
            if index not in final_results_withcomp.keys():
                final_results_wotext.append(comp[index])
                cv2.drawContours(final_graph_im, [comp[index]], 0, (0, 0, 255), 2)
        
        cv2.imshow("final_graph_im", final_graph_im)
        cv2.waitKey(0)   
            
        return (final_results_withcomp, final_results_wocomp)
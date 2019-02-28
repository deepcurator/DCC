# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:01:40 2019

@author: Amar Viswanathan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 13:50:13 2019

@author: Amar Viswanathan
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
import re




#root_dir = 'C:\\Users\\z003z47y\\Documents\\gitlab\\CognitiveAutomationEngineering\\examples\\TIAPortal2CEG\\20180725_Export\\'

root_dir = 'C:\\Users\\z003z47y\\Documents\\git\\darpa_aske_dcc\\src\\paperswithcode\\data\\'
textIterator = glob.glob(root_dir + '**/*.pdf', recursive=True)
index = 0
txtfileNames = []

for file in textIterator:
   startIndex = textIterator[index].rfind('\\')
   endIndex = textIterator[index].rfind('.')
   fileName = textIterator[index][startIndex+1:endIndex]
   
   
   fileNameString = textIterator[index][0:startIndex] + '\\' + fileName + '.txt'
   txtfileNames.append(fileNameString + "::" + textIterator[index][0:startIndex] + '\\' +'title.txt' )
   index += 1

index = 0

for fileName in txtfileNames:
    fileAndTitle = fileName.split('::')
    print(fileAndTitle[0] + ";" + fileAndTitle[1])
    with open(fileAndTitle[0], "r", encoding = 'utf8') as a, open(fileAndTitle[1], "r",encoding='utf8') as b:
            lines = a.readlines()
            title = b.readline()
            title = str(title)
            title.strip()
#            print(title)
            with open('C:\\Users\\z003z47y\\Documents\\git\\darpa_aske_dcc\\src\\paperswithcode\\txt\\papersv2\\' + str(index) + '.txt','a',encoding='utf8') as f1:
                f1.write('Title : ' + str(title) + "\n")
                f1.write(''.join(str(line) for line in lines))
            f1.close()
    a.close()
    b.close()
    index +=1    
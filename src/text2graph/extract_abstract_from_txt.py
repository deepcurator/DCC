# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:12:38 2019

@author: Amar Viswanathan
"""

import re
import glob

root_dir = 'C:\\Users\\z003z47y\\Documents\\git\\darpa_aske_dcc\\src\\paperswithcode\\txt\\papers'
textIterator = glob.glob(root_dir + '**/*.txt', recursive=True)

abstracts = []

start  = 'abstract'
end = 'introduction'


for file in textIterator:
   print(file)
   with open(file,encoding='utf8') as f:
       lines = f.readlines()
       papertxt = ''.join(str(line) for line in lines)
       papertxt = papertxt.lower()
       result = papertxt.split(start)[1].split(end)[0]
       abstracts.append(result)
       
       
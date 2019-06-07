# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:12:38 2019

@author: Amar Viswanathan
"""
import os
import yaml
import glob


start  = 'abstract'
end = 'introduction'

config = yaml.safe_load(open('../../conf/conf.yaml'))
root_dir = config['EXTRACT_TEXT_PATH']
out_dir = config['EXTRACT_ABSTRACT_PATH']
textIterator = glob.glob(root_dir + '*.txt', recursive=True)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
for file in textIterator:
   print(file)
   with open(os.path.join(root_dir,file),encoding='utf8') as f:
       lines = f.readlines()
       papertxt = ''.join(str(line) for line in lines)
       papertxt = papertxt.lower()
       result = papertxt.split(start)[1].split(end)[0]
       out = os.path.join(out_dir,file)
       file = open(out,'w', encoding = 'utf8') 
       file.write(result) 
       file.close() 
       
       
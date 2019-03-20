# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:03:04 2019

@author: dfradkin
"""
import os
import pandas as pd
from nltk import tokenize
import re

# import nltk
# nltk.download('punkt')

# inpout directory: has to have .txt and .ann files
#p='C:\\Users\\dfradkin\\Desktop\\abstracts'
p='C:\\Home\src\\Python\\ASKE\\devASKE\\Data\\abstracts-annotated'
# output directory: will have files i-j.txt and i-j.ann where 'i' is index of abstract
# and 'j' is index of sentence
out_dir='C:\\Home\src\\Python\\ASKE\\devASKE\\Output\\BreakBrat\\abstracts-annotated'

text_pattern= re.compile('(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])')

def split_annotations(text, ann, out_dir, i):
    '''
    text - input text
    ann - old annotations
    i - id of the file
    '''    
    # identify sentences
    sent=tokenize.sent_tokenize(text)     
    ### go through sentences, keep track of types/relation in it:
    for (j, s) in enumerate(sent):
        # check that it is a legitimate sentence
        if len(sent)<5 or  re.search('[a-zA-Z]',s) is None:
            continue        
        # save sentence to file
        sent_file=os.path.join(out_dir,str(i)+'-'+str(j)+".txt") 
        fh=open(sent_file,"w",encoding='utf8',errors='ignore'); fh.write(s); fh.close();
        # get sentence range:
        sentence_start=text.find(s)
        sentence_end=sentence_start+len(s)
        # find annotations for this sentence:
        entities=dict() # entities in this sentence:
        small_ann=[]
        for row_num in range(len(ann)):
            ne_type=ann.values[row_num,0]
            if ne_type.startswith('T'):
                ne_pos=ann.values[row_num,1].split(' ')
                if int(ne_pos[1])>= sentence_start and int(ne_pos[1])<=sentence_end:
                    entities[ne_type]=1
                    # create new entry:
                    middle=' '.join([ne_pos[0],str(int(ne_pos[1])-sentence_start),str(int(ne_pos[2])-sentence_start)])
                    new_entry=[ne_type,middle ,ann.values[row_num,2]]                    
                    small_ann.append(new_entry)
            else:   # relation:
                rel=ann.values[row_num,1].split(' ')
                if rel[1] in entities and rel[2] in entities:
                    record=[ann.values[row_num,0],ann.values[row_num,1],ann.values[row_num,2]]
                    if pd.isna(record[2]):
                        record[2]=''
                    small_ann.append(record)
                    row_num=row_num+1
        # save new ann file:
        ann_file=os.path.join(out_dir,str(i)+'-'+str(j)+".ann") 
        with open(ann_file, 'w',encoding='utf8',errors='ignore') as f:
            for record in small_ann:
                f.writelines('\t'.join(record)+'\n')



for i in range(96):
    # load text:
    print("processing file: ", i)
    txt_file=os.path.join(p,str(i)+".txt")
    if not os.path.exists(txt_file):
        continue
    fh = open(txt_file, "r",encoding='utf8',errors='ignore')
    text = fh.read()
    fh.close()    
    # load ann file
    ann_file=os.path.join(p,str(i)+".ann")  
    if os.stat(ann_file).st_size==0:
        ann=[]
    else:
        ann=pd.read_csv(ann_file, delimiter='\t', header=None, names=None)
    # split annotations:
    split_annotations(text, ann, out_dir, i)

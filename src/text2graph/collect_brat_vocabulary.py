# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:30:38 2019

@author: Dmitriy Fradkin
"""
import numpy as np
import os
import yaml
import glob
import pandas as pd
import pickle
import re

'''
x is a row of brat annotation
'''
def create_relation(x, entity_dict):
    rel=x[1].split(' ')
    if x[0]=='*':
        tup=((entity_dict[rel[1]],entity_dict[rel[2]]),rel[0])
    else:   
        e1=rel[1][5:]
        e2=rel[2][5:]
        tup=((entity_dict[e1],entity_dict[e2]),rel[0])
    return( tup )
        
toskip=['^a','^the','novel', 'our', 'proposed', 'previous', 'state-of-the-art', 'their','new']
def preprocess_ent(text):
    # remove articles and white spaces:
    for t in toskip:
        text= re.sub(r"\b"+t+r"\b", "",text)
        #print(text)
    text=text.replace(r"\s+"," ").strip().lower()
    return(text)

def collect_annotations(textIterator):
    entities=[]
    relations=[]
    vocab=dict()
    rel_vocab=dict()
    for ann_file in textIterator:
       print(ann_file)
       if os.stat(ann_file).st_size == 0:
           continue
       ann=pd.read_csv(ann_file, delimiter='\t', header=None, names=None)
       ann.columns=['shortid','type_pos','text']
       # create mapping of strings to entities 
       ann2=ann[ann.shortid.apply(lambda x: x.startswith('T'))].copy()
       ann2['text']=ann2['text'].apply(preprocess_ent).values
       #d=dict(ann2.apply(lambda x: (x['text'],x['type_pos'].split(' ')[0]),axis=1).values)
       ent_list=ann2.apply(lambda x: (x['text'],x['type_pos'].split(' ')[0]),axis=1).values
       # skip empty strings (may happen after cleaning)
       ent_list=[x for x in ent_list if len(x[0])>0]
       #vocab.update(dict(ent_list))
       entities.extend(ent_list)
       ent_map=dict(zip(ann2['shortid'].values,ann2['text'].values))
       # create relations mapping:
       ann3=ann[ann.shortid.apply(lambda x: not(x.startswith('T')))]
       #rel_d=dict(ann3.apply(lambda x: create_relation(x,ent_map),axis=1).values)
       rel_list=ann3.apply(lambda x: create_relation(x,ent_map),axis=1).values
       #rel_vocab.update(dict(rel_list))
       relations.extend(rel_list)
    entities2=sorted(set(entities)) 
    relations2=sorted(set(relations))
    return(vocab, rel_vocab, entities2,relations2)

###########################################
            
if __name__ == '__main__':

    config = yaml.safe_load(open('../../conf/conf.yaml'))
    annot_dir = config['ANNOTATED_TEXT_PATH']    
    textIterator = glob.glob(annot_dir + '*.ann', recursive=True)
    (vocab, rel_vocab, entities2,relations2)= collect_annotations(textIterator)

    #save results
    outdir=config['MODEL_PATH']
    f = open(os.path.join(outdir,'dictionaries.pcl'), 'wb')
    pickle.dump([vocab,rel_vocab],f)
    f.close()
    
    f = os.path.join(outdir,'brat_entities.txt')
    with open(f, 'w',encoding='utf8') as writeFile:
        for term,ent in entities2:
            writeFile.writelines(term+'\t'+ent+'\n')
    writeFile.close()
    
    f = os.path.join(outdir,'brat_relations.txt')
    with open(f, 'w',encoding='utf8') as writeFile:
        for term,ent in relations2:
            writeFile.writelines(term[0]+'--'+term[1]+'\t'+ent+'\n')
    writeFile.close()



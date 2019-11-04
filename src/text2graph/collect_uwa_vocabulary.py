# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:30:38 2019

@author: Dmitriy Fradkin
"""
import numpy as np
import os
import yaml
import glob
import pickle
import pandas as pd
from collect_brat_vocabulary import collect_annotations


def remap(x):
    if x=='Metric':
        return('Eval')
    if x=='OtherScientificTerm':
        return('Other')
    if x=='HYPONYM-OF':
        return('isA')
    x=x.lower().capitalize()
    return(x)
    

###########################################
            
if __name__ == '__main__':

    config = yaml.safe_load(open('../../conf/conf.yaml'))
    uwa_dir = 'C:\\home\\projects\\DARPA ASKE\\sciERC_raw\\raw_data\\'   
    textIterator = glob.glob(uwa_dir + '*.ann', recursive=True)
    (vocab, rel_vocab, entities2,relations2)= collect_annotations(textIterator)

    
    ents=pd.DataFrame(entities2,columns=['text','ent_type']).drop_duplicates()
    # remap to our entities
    ents.ent_type=ents.ent_type.map(remap)
    # avoid internal conflicts: remove entities and rel' with multiple types...
    ents2=ents.groupby(by='text').filter(lambda x: len(x) == 1)

    #gr=ents.groupby(by='text')
    #for nm,g in gr:
    #    if len(g)>1:
    #        print('conflict {}: {} vs {}'.format(nm,g.ent_type.iloc[0],g.ent_type.iloc[1]))

    relations3=[('--'.join(x[0]),x[1]) for x in relations2]
    rels=pd.DataFrame(relations3,columns=['pair','rel_type']).drop_duplicates()
    # remap to our entities
    rels.rel_type=rels.rel_type.map(remap)  
    # remove 'Coref'
    rels=rels[rels.rel_type!='Coref']
    # avoid internal conflicts: remove entities and rel' with multiple types...
    rels2=rels.groupby(by='pair').filter(lambda x: len(x) == 1)

    #save results
    outdir=config['MODEL_PATH']
    f = open(os.path.join(outdir,'uwa_dictionaries.pcl'), 'wb')
    pickle.dump([vocab,rel_vocab],f)
    f.close()
    
    f = os.path.join(outdir,'uwa_brat_entities.txt')
    ents2.to_csv(f,index=False,header=False, sep='\t')
    
    f = os.path.join(outdir,'uwa_brat_relations.txt')
    rels2.to_csv(f,index=False,header=False, sep='\t')    



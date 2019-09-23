# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:31:33 2019

@author: dfradkin
"""

import os 
from xml.dom import minidom
import yaml
import glob
from auto_annotate import TextAnnotator
import pickle
       
def getFigDesc(fig):
    for node in fig.childNodes:
        if node.tagName == "figDesc":
            if len(node.childNodes) > 0:
                return node.childNodes[0].data
            return ""

###########################################
            
if __name__ == '__main__':
    ################## Load auto-annotate: ############
    config = yaml.safe_load(open('../../conf/conf.yaml'))
    model_dir=config['MODEL_PATH']
    f = open(os.path.join(model_dir,'full_annotations.pcl'), 'rb')
    [entity_map,uri2entity, uri2rel]=pickle.load(f)
    f.close()
    annotator=TextAnnotator(entity_map,uri2entity, uri2rel)        
            
    ################## Start Processing xml: ############        
    #config = yaml.safe_load(open('../../conf/conf.yaml'))
    #root_dir = config['EXTRACT_TEXT_PATH'] 
    root_dir='../../../grobid/'
    #out_dir = config['EXTRACT_ABSTRACT_PATH']
    textIterator = glob.glob(root_dir + '*.xml', recursive=True)
    
    count = 0
    count_fig = 0
    for file in textIterator:
        fn=os.path.basename(file)
        paper_name=fn.replace('.tei.xml','')
        print(fn)
        cap_fn=os.path.join('../../Data/extracted_captions/',paper_name+'.captions')
        with open(cap_fn,'w',encoding='utf8') as cap_fp: 
            xmldoc= minidom.parse(file)
            allFig = xmldoc.getElementsByTagName('figure')
            for fig in allFig:
                figure_uid='Figure_'+str(count_fig)
                fid=fig.getAttribute('xml:id')
                ft=fig.getAttribute('xml:type')
                if len(ft)==0: 
                    ft='figure'
                cap_fp.write(paper_name+'\thasFigure\t'+figure_uid+'\n')
                cap_fp.write(figure_uid+'\thasFigureId\t'+fid+'\n')      
                cap_fp.write(figure_uid+'\thasFigureType\t'+ft+'\n')
                for node in fig.childNodes:
                    if node.tagName == "head":
                        if len(node.childNodes) == 0:
                            pass
                        else:
                            head = node.childNodes[0].data
                            if "fig." not in head and "Fig." not in head and "figure" not in head and "Figure" not in head:
                                count += 1
                            else:
                                cap_fp.write(figure_uid+'\thasHead\t'+head+'\n')
                                figDesc = getFigDesc(fig)
                                # output caption text
                                text_fn=os.path.join('../../Data/extracted_captions/',paper_name+'.'+fid+'.txt')
                                with open(text_fn,'w',encoding='utf8') as txt_fp:
                                    txt_fp.write(figDesc)
                                # output caption annotations
                                ann_triples=annotator.match_terms(figDesc)    
                                if len(ann_triples)>0:
                                    ann_fn=os.path.join('../../Data/extracted_captions/',paper_name+'.'+fid+'.ann')
                                    with open(ann_fn,'w',encoding='utf8') as ann_fp: 
                                        for ann in ann_triples:
                                            # note entities: (ignore relations for now?)
                                            if len(ann)==3:
                                                cap_fp.write(figure_uid+'\tcaptionContains\t'+ann[2]+'\n')
                                            ann_fp.write('\t'.join(ann)+'\n')
                                        cap_fp.write('\n')
                                        ann_fp.write('\n')
                                count_fig += 1
    print(count)
    print(count_fig)       
       
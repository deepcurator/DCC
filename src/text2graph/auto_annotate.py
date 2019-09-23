import os
import re
import yaml 
import pickle 
import itertools
import pandas as pd
#from nltk.corpus import stopwords
#en_stopwords = stopwords.words('english')


class TextAnnotator:
    def __init__(self, entity_map,uri2entity, uri2rel):
        self.entity_map=entity_map
        self.uri2entity=uri2entity
        self.uri2rel=uri2rel

    ########## Generate annotations for text #########
    def match_terms(self,text):        
        # ensure text is lower case
        text=text.lower()        
        countT=1
        anns=[]
        # list matched entities - note that countT will be shifted by 1
        entity_matches=[]
        for (ent, uri) in self.entity_map.items():
            # use regular expressions - match full words (possibly with plural at the end)
            regex=re.compile('\W'+re.escape(ent)+'s?\W')
            for m in regex.finditer(text):
                start=m.start()+1
                end=m.end()-1
                # known entity
                if uri in self.uri2entity:
                    ent_type=self.uri2entity[uri]
                    anns.append([countT,start,end,text[start:end],len(ent),uri,ent_type])
                    entity_matches.append((countT,uri))
                    countT=countT+1
        # generate ann triples for entities: sort by start, end and length of entity
        ann_df=pd.DataFrame(anns,columns=['ID','SP','EP','TEXT','LEN','URI','ENT_TYPE']).sort_values(by=['SP','EP','LEN'])
        ann_df=TextAnnotator.clean_annotations(ann_df)
        # create annotation tuples and get map of entity matches
        ann_triples=ann_df.apply(lambda row: ('T'+str(row.ID),'{} {} {}'.format(row.ENT_TYPE,row.SP,row.EP), row.TEXT), axis=1).values.tolist()    
        entity_matches=ann_df.apply(lambda row: (row.ID,row.URI), axis=1).values.tolist()
        
        countR=1
        # generate all candidate pairs:
        pair_list=itertools.combinations(entity_matches, 2)
        for ent_pair in pair_list:
            x=ent_pair[0]
            y=ent_pair[1]
            if x[0]==y[0]:
                continue            
            uri1=x[1]
            uri2=y[1]                        
            if uri1==uri2:
                if x[0]<y[0]:
                    triple=('*','{} T{} T{}'.format('sameAs',x[0],y[0]))
                    ann_triples.append(triple)
            elif (uri1,uri2) in self.uri2rel: 
                r=self.uri2rel[(uri1,uri2)]
                triple=()
                if r=='sameAs':
                    triple=('*','{} T{} T{}'.format(r,x[0],y[0]))
                else:
                    triple=('R'+str(countR),'{} Arg1:T{} Arg2:T{}'.format(r,x[0],y[0]))
                    countR=countR+1
                #print(triple)
                ann_triples.append(triple)
        return(ann_triples)

    '''
    Assume data frame is sorted by start position, end position and length of entity
    '''
    @staticmethod
    def clean_annotations(ann_df):
        ids_to_remove=[]
        for i in range(len(ann_df)):
            row1=ann_df.iloc[i]
            if row1.ID not in ids_to_remove:
                for j in range(i+1,len(ann_df)):
                    row2=ann_df.iloc[j]
                    if row2.SP >= row1.EP:
                        # no overlap between rows
                        break
                    if row2.SP == row1.SP:
                        # skip row1 - it is inside row2
                        ids_to_remove.append(row1.ID)
                        break
                    if row2.EP <= row1.EP:
                        # row2 is within row1 (row1 is not covered by any other)
                        # but keep going, there may be more like this
                        ids_to_remove.append(row2.ID)
        ann_df = ann_df[~ann_df['ID'].isin(ids_to_remove)]
        return(ann_df)

    

if __name__ == '__main__':
    ################## Load entity and relation maps: ############
    config = yaml.safe_load(open('../../conf/conf.yaml'))
    model_dir=config['MODEL_PATH']
    
    f = open(os.path.join(model_dir,'full_annotations.pcl'), 'rb')
    [entity_map,uri2entity, uri2rel]=pickle.load(f)
    f.close()
    annotator=TextAnnotator(entity_map,uri2entity, uri2rel)
    
    file_path=config['SENTENCE_ANNOTATED_TEXT_PATH']
    
    ################## Process multiple files ############
    
    files=[x for x in os.listdir(file_path) if os.path.splitext(x)[1]=='.txt']
    for f in files:
        with open(os.path.join(file_path,f),encoding='utf8') as fp:
           text = fp.read().strip()
        print('Processing '+f)
        ann_triples=annotator.match_terms(text)
        # output to .ann file:
        outfile=os.path.join(file_path,f).replace('.txt','.ann2')
        with open(outfile,'w',encoding='utf8') as fp:
            for ann in ann_triples:
                fp.write('\t'.join(ann)+'\n')
                
    
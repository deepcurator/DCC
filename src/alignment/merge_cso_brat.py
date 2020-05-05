import os
import yaml 
import pickle 
import rdflib 
import urllib.parse
import pandas as pd
#from nltk.corpus import stopwords
#en_stopwords = stopwords.words('english')

################## Load graph and prepare structures for matching ############
config = yaml.safe_load(open('../../conf/conf.yaml'))
model_dir=config['MODEL_PATH']

### load cso:
f = open(os.path.join(model_dir,'cso_dict.pcl'), 'rb')
[cso_entity_map,cso_relation_map]=pickle.load(f)
f.close()

### load and merge our brat and UWA annotations:
def merge_annot(annot1, annot2):
    out_df=pd.concat([annot1, annot2], axis=0).drop_duplicates()
    out_df2=out_df.groupby(by=out_df.columns[0]).filter(lambda x: len(x) == 1)
    return(out_df2)
    #gr=ents.groupby(by='text')
    #for nm,g in gr:
    #    if len(g)>1:
    #        print('conflict {}: {} vs {}'.format(nm,g.ent_type.iloc[0],g.ent_type.iloc[1]))

# note: we load from manually cleaned files
fn=os.path.join(model_dir,'brat_entities_clean.txt')
brat_entities=pd.read_csv(fn, sep='\t', header=None)
brat_entities.columns=['text','ent_type']

fn=os.path.join(model_dir,'brat_relations_clean.txt')
brat_relations=pd.read_csv(fn, sep='\t', header=None)
brat_relations.columns=['pair','rel_type']

# load UWA annotationss
fn=os.path.join(model_dir,'uwa_brat_entities.txt')
uwa_entities=pd.read_csv(fn, sep='\t', header=None)
uwa_entities.columns=['text','ent_type']

fn=os.path.join(model_dir,'uwa_brat_relations.txt')
uwa_relations=pd.read_csv(fn, sep='\t', header=None)
uwa_relations.columns=['pair','rel_type']


ent_df=merge_annot(brat_entities,uwa_entities)
rel_df=merge_annot(brat_relations,uwa_relations)

vocab=dict(zip(ent_df.text,ent_df.ent_type))
### manual corrections:
vocab['hses']='Other'
ps=rel_df.pair.apply(lambda x: tuple(x.split('--'))).values
rel_vocab=dict(zip(ps,rel_df.rel_type))



### generate connected components from brat annotations:
import networkx as nx
def graph_conversion(rel_vocab):
    G = nx.Graph()
    for (pair,rel) in rel_vocab.items():
        if rel!='sameAs':
            continue
        for n in pair:
            G.add_node(n)
        G.add_edge(pair[0],pair[1])
    return(G)

G=graph_conversion(rel_vocab)
# connected components:
cc=[c for c in nx.connected_components(G)]
# add these to cso_entity_map with existing or new uri:
for comp_set in cc:
    comp=list(comp_set)
    uri=''
    # check if there is an existing URI for the component:
    for term in comp:
        if term in cso_entity_map:
            uri=cso_entity_map[term]
            break
    # component doesn't have a URI: likely because there are no brat entities corresponding to relation in question
    if uri=='':
        #print('Component of size {} not mapped'.format(len(comp)))
        continue
    # add uri for all terms:
    for term in comp:
        if term not in cso_entity_map:
            cso_entity_map[term]=uri
            #print('Added(1): "{}": {}'.format(term,uri))
        elif uri!=cso_entity_map[term]:
            print('Conflict(1): "{}": {} vs {}'.format(term, uri,cso_entity_map[term]))


### try to link cso strings/topics to annotated entities:
# for each CSO string/entity, check for match in brat annotation
uri2entity={}
cso_topic_list=[x for x in cso_entity_map.keys()]
for topic in cso_topic_list:
    # found entity for cso topic - link uri to entity from annotations
    if topic in vocab:
        uri=cso_entity_map[topic]
        if uri not in uri2entity:
            uri2entity[uri]=vocab[topic]            
        elif uri2entity[uri]!=vocab[topic]:
            # conflict:
            print('Conflict(2) "{}": {} vs {}'. format(topic,uri2entity[uri], vocab[topic]))
                        
# for each term in annotations, if there is still no match in cso, create a new uri 
# and add to uri2entity            
for term,entity in vocab.items():
    if term not in cso_entity_map:
        # create uri
        # uri=rdflib.URIRef('https://siemens.com/'+urllib.parse.quote(term.replace(' ','_')))
        uri=rdflib.URIRef('https://github.com/deepcurator/DCC/'+urllib.parse.quote(term.replace(' ','_')))
        cso_entity_map[term]=uri
        # add to 
        uri2entity[uri]=entity

# convert all relations (except sameAs) from annotations to uri relations            
uri2rel={}
for pair,rel in rel_vocab.items():
    if rel!='sameAs':
        # if entities not in vocab, ignore rel:
        if pair[0] not in cso_entity_map or pair[1] not in cso_entity_map:
            #print('Skipping "{}"-"{}": {}'. format(pair[0],pair[1],rel))
            continue
        uri1=cso_entity_map[pair[0]]
        uri2=cso_entity_map[pair[1]]
        if (uri1,uri2) not in uri2rel:
            uri2rel[(uri1,uri2)]=rel
        elif rel!=uri2rel[(uri1,uri2)]:  
            print(pair)            
            print('Conflict(3) for "{}"-"{}": {} vs {}'. format(uri1,uri2,rel, uri2rel[(uri1,uri2)]))

# add rels from cso:
for pair, rel in cso_relation_map.items():
    if pair in uri2rel and uri2rel[pair]!=rel:
        print('Conflict(4) for "{}"-"{}": {} vs {}'. format(pair[0],pair[1],rel, uri2rel[pair]))
    else:
         uri2rel[pair]=rel
        
        
### Save these structures:
outdir=config['MODEL_PATH']
f = open(os.path.join(outdir,'full_annotations.pcl'), 'wb')
pickle.dump([cso_entity_map,uri2entity, uri2rel],f)
f.close()


import os
import re
import numpy as np
import rdflib
from rdflib.plugins.sparql import prepareQuery
import urllib
import yaml 
import pickle 
from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')

################## Load graph and prepare structures for matching ############
config = yaml.safe_load(open('../../conf/conf.yaml'))
cso_dir = config['CSO_PATH'] #''

g = rdflib.Graph()
g.load(os.path.join(cso_dir,"CSO.3.1.owl"))

queryInheritance = prepareQuery("ask {?entity1 cso:superTopicOf* ?entity2 . }",
                         initNs={'cso':'http://cso.kmi.open.ac.uk/schema/cso#'})
def check_SuperTopicOf(g,e1,e2):
    qres=g.query(queryInheritance, initBindings={'entity1': rdflib.URIRef(e1),'entity2': rdflib.URIRef(e2)})
    out=[x for x in qres]
    return(out[0])


relation_translate={}
relation_translate['cso#contributesto']='Used-for'
relation_translate['cso#relatedequivalent']='sameAs'
relation_translate['cso#supertopicof']='isA'
relation_translate['cso#preferentialequivalent']='sameAs'

# note: assumes everything is same case
def is_abbrev(abbrev, text):
    if(len(abbrev)>=len(text)):
        return False
    # continuous letter matches starting from spaces
    pattern = "(|.*\W)".join(abbrev)
    # alternative: any subset of letters:
    # pattern = ".*".join(abbrev)
    flag=re.match("^" + pattern, text) is not None
    # check if removing 's' at end of abbrev would work:
    if abbrev.endswith('s'):
        flag=is_abbrev(abbrev[:-1], text)
    return flag

abbr_re=re.compile('\(\w+\)')
def process_item(e):
    parts=str(e).split('/')
    if parts[2]=='cso.kmi.open.ac.uk':
        # decode to handle special characters in terms
        term=urllib.parse.unquote(parts[-1]).replace('_',' ').strip().lower()
        # separate abbreviations: but there are many other issues...
        match=abbr_re.search(term)
        if match is None:
            return([term])
        ### check if the match corresponds to abbreviation of what came before or vice-versa:
        start=match.start()
        end=match.end()
        outer_term=term[0:start-1].strip()
        if end<len(term)-1:
            outer_term=outer_term+term[end+1:].strip()
        inner_term=term[start+1:end-1].strip()
        if is_abbrev(inner_term,outer_term) or is_abbrev(outer_term, inner_term):
            res=[outer_term,inner_term]
            res=[x for x in res if x not in en_stopwords and len(x)>1]
            return(res)
        #else:
        #    print(inner_term+' --- '+outer_term+' --- False')
    return None        


### 1st pass through graph:
# create set of all entities, create preference maps
# by default, entities are their own prefered equivalents
entities={} # map uris to strings
relations={}
preference_map={}
equivalence_sets={}
for s,p,o in g:
    if s not in entities:
        entities[s]=process_item(s)
        if entities[s] is not None and s not in preference_map:
            preference_map[s]=s
    if o not in entities:
        entities[o]=process_item(o)
        if entities[o] is not None and o not in preference_map:
            preference_map[o]=o
    if p not in relations:
        relations[p]=process_item(p)
    if relations[p] is not None and relations[p][0]=='cso#preferentialequivalent':
        preference_map[s]=o    

### 2nd pass:
# replace all values in entity maps with preferred entities from preference_map
# extract relations (use preferred entities)
# create equivalence sets of strings for a uri!!!
problems=[]
entity_map={}
relation_map={}
for rel in g:
    s=entities[rel[0]]
    o=entities[rel[2]]
    pairs=[(s,0), (o,2)]
    for (t,ind) in pairs:     
        if t is not None:
            for k in t:
                if k in entity_map and entity_map[k]!=rel[ind]:
                    problems.append([k,entity_map[k],rel[ind]])
                entity_map[k]=preference_map[rel[ind]]
                if entity_map[k] not in equivalence_sets:
                    equivalence_sets[entity_map[k]]=set()
                equivalence_sets[entity_map[k]].add(k)
    p=relations[rel[1]]
    if p is not None and p[0] in relation_translate.keys() and s is not None and o is not None:
        if p[0]=='cso#supertopicof':
            # need to switch 's' and 'o'
            relation_map[(preference_map[rel[2]],preference_map[rel[0]])]=relation_translate[p[0]]
        else: # if p[0] is 'cso#contributesTo':            
            relation_map[(preference_map[rel[0]],preference_map[rel[2]])]=relation_translate[p[0]]

### save entity and relation maps:
outdir=config['MODEL_PATH']
f = open(os.path.join(outdir,'cso_dict.pcl'), 'wb')
pickle.dump([entity_map,relation_map, equivalence_sets],f)
f.close()

### Output all terms from CSO        
ent_list=[x for x in entity_map.keys()]
f = os.path.join(outdir,'cso_terms.txt')
with open(f, 'w') as writeFile:
    for x in sorted(ent_list):
        writeFile.writelines(x+'\n')
writeFile.close()

### Output all topics that need to be labelled:
pref_topics=np.unique([x for x in preference_map.values()])
f = os.path.join(outdir,'cso_preferred_topics.txt')
with open(f, 'w',encoding='utf8') as writeFile:
    for x in sorted(pref_topics):
        writeFile.writelines(x+'\n')
writeFile.close()


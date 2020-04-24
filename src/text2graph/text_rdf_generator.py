from __future__ import unicode_literals, print_function
from os import listdir
from os.path import isfile, join

# RDF specific libraries
from rdflib import URIRef, Literal
from rdflib.namespace import RDF 
from rdflib import Graph

# spacy specific libraries
import spacy

#nltk libraries
import nltk

# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def getabstract(filename):
    abstxt = ""
    with open(filename, encoding="utf8") as file:
        lines = file.readlines()
        abstxt = ''.join(str(line) for line in lines)
        abstxt = abstxt.lower()
    return abstxt


def createRDF(filename, entity_map, consolidatedGraph=None, model_dir='', text_dir=''):
    # g = Graph()
    # g.parse(ontology,format="n3")
    dcc_namespace = "https://github.com/deepcurator/DCC/"
    
    # print(filename)
    textfilename = text_dir + filename
    # print("Processing: " + textfilename)
    # print(getabstract(textfilename))

    ## filename will act as a unique URI to connect all the three graphs

    filesubject = dcc_namespace + filename
    ### DF added:
    triple_list = [] 
    # g.add((URIRef(filesubject),RDF.type,URIRef(dcc_namespace + "Publication")))
    # consolidatedGraph
    if consolidatedGraph is not None:
        consolidatedGraph.add((URIRef(filesubject),RDF.type,URIRef(dcc_namespace + "Publication")))
    
    #load the spacy nlp model
    nlp = spacy.load(model_dir)
    sents = nltk.sent_tokenize(getabstract(textfilename))
    entity_dict = {}
    for sentence in sents:
        ner_tagged = nlp(sentence)
        tagged_entities = ner_tagged.ents   
        for entity in tagged_entities:
            # print(entity.text, entity.label_)
            if entity.text not in entity_dict:
                entity_dict[entity.text] = entity.label_

    for entitytext, entitylabel in entity_dict.items():
        entitytext = entitytext.replace(" ",'_')
        if(entitytext in entity_map):
            csovalue = entity_map[entitytext]
            str_value = str(csovalue)
            if("cso" in str_value):
                consolidatedGraph.add((URIRef(filesubject + "_" + entitytext),URIRef(dcc_namespace + "hasCSOEquivalent"),csovalue))
        # print(entitytext)
        # print(filesubject + "_" + entitytext)
        if consolidatedGraph is not None:
            consolidatedGraph.add((URIRef(filesubject + "_" + entitytext),RDF.type,URIRef(dcc_namespace + entitylabel)))
            consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "hasEntity"),URIRef(filesubject + "_" + entitytext)))
            textLiteral = Literal(entitytext)
            consolidatedGraph.add((URIRef(filesubject + "_" + entitytext),URIRef(dcc_namespace + 'hasText'),textLiteral))

        ### DF added:
        triple_list.append([entitytext, "is_a", entitylabel])
        triple_list.append([filename.replace(" ", "_"), "has_entity", entitytext])

    print("Completed processing file " + filename) 
    return(triple_list)
 
def save_triples(triple_list, triplesFile):
    with open(triplesFile, 'w') as f:
        for triple in triple_list:
            f.write(triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\n")

def createTextRDF(text_dir, destinationfolder, ontology, model_dir):    
    # Ontology locations
    # ontology = "/home/z003z47y/git/DCC/src/ontology/DeepSciKG.nt"

    consolidatedGraph = Graph() 
    # consolidatedGraph.parse(ontology, format="n3") 
    
    # model_dir = 'Models'

    # mypath = "C:/aske-2/dcc/grobid-workspace/output"
    onlyFiles = [f for f in listdir(text_dir) if isfile(join(text_dir, f))]
    
    # iterate through the rows in the dataframe
    #  for index,row in df.iterrows():
    for f in onlyFiles:
       if f.endswith(".txt"):
           _=createRDF(f, consolidatedGraph, model_dir, text_dir)
    
    destinationfile = destinationfolder + "text2graph.ttl"
    print("Saving rdf file " + destinationfile)
    consolidatedGraph.serialize(destination=destinationfile,format='turtle')

    
def createTextTriples(text_dir, destinationfolder, ontology, model_dir):    
    onlyFiles = [f for f in listdir(text_dir) if isfile(join(text_dir, f))]    
    # iterate through the rows in the dataframe
    #  for index,row in df.iterrows():
    for f in onlyFiles:
       if f.endswith(".txt"):
           triple_list = createRDF(f, None, model_dir, text_dir)
           triplesFile = destinationfolder + "text2graph.triples"
           save_triples(triple_list, triplesFile)
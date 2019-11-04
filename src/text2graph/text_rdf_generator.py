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
    with open(filename) as file:
        lines = file.readlines()
        abstxt = ''.join(str(line) for line in lines)
        abstxt = abstxt.lower()
    return abstxt


def createrdf(filename, consolidatedGraph, model_dir, text_dir):
    # g = Graph()
    # g.parse(ontology,format="n3")
    dcc_namespace = "https://github.com/deepcurator/DCC/"
    
    # print(filename)
    textfilename = text_dir + filename
    # print("Processing: " + textfilename)
    # print(getabstract(textfilename))

    ## filename will act as a unique URI to connect all the three graphs

    filesubject = dcc_namespace + filename
    # g.add((URIRef(filesubject),RDF.type,URIRef(dcc_namespace + "Publication")))
    # consolidatedGraph
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
        # print(entitytext)
        # print(filesubject + "_" + entitytext)
        consolidatedGraph.add((URIRef(filesubject + "_" + entitytext),RDF.type,URIRef(dcc_namespace + entitylabel)))
        consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "hasEntity"),URIRef(filesubject + "_" + entitytext)))
        textLiteral = Literal(entitytext)
        consolidatedGraph.add((URIRef(filesubject + "_" + entitytext),URIRef(dcc_namespace + 'hasText'),textLiteral))

    print("Completed processing file " + filename)
 

def createTextRDF(text_dir, destinationfolder, ontology, model_dir):    
    # Ontology locations
    # ontology = "/home/z003z47y/git/DCC/src/ontology/DeepSciKG.nt"

    consolidatedGraph = Graph() 
    consolidatedGraph.parse(ontology, format="n3") 
    
    # model_dir = 'Models'

    # mypath = "C:/aske-2/dcc/grobid-workspace/output"
    onlyFiles = [f for f in listdir(text_dir) if isfile(join(text_dir, f))]
    
    # iterate through the rows in the dataframe
    #  for index,row in df.iterrows():
    for f in onlyFiles:
       if f.endswith(".txt"):
           createrdf(f, consolidatedGraph, model_dir, text_dir)
    
    destinationfile = destinationfolder + "text2graph.ttl"
    print("Saving rdf file " + destinationfile)
    consolidatedGraph.serialize(destination=destinationfile,format='turtle')
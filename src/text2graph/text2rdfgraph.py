from __future__ import unicode_literals, print_function
import os
from pathlib import Path
import glob
import pandas as pd
import yaml
import pickle

# RDF specific libraries
from rdflib import URIRef, BNode, Literal
from rdflib.namespace import RDF, FOAF 
from rdflib import Graph

# spacy specific libraries
import spacy
#import plac

#nltk libraries
import nltk


def getabstract(filename):
    abstxt = ""
    with open(filename) as file:
        lines = file.readlines()
        abstxt = ''.join(str(line) for line in lines)
        abstxt = abstxt.lower()
    return abstxt

def save_triple_file(triple_list,filename):
    triplefilename = triple_dir + filename + "/t2g.triples"
    if not os.path.exists(triple_dir + filename):
        os.makedirs(triple_dir + filename)
    with open(triplefilename,'w+') as f :
        for line in triple_list:
            f.write(line + "\n")

def createrdf(row,csomap, consolidatedGraph):

    triple_list = []

    dcc_namespace = "https://github.com/deepcurator/DCC/"

    # print(row['paper_title'],row['paper_link'],row['conference'], row['year'], row['Platform'])
    filename = row['paper_link'].split('/')[-1]
    if(filename.endswith('.pdf')):
        filename = filename.split('.pdf')[0]
    elif(filename.endswith('.html')):
        filename = filename.split('.html')[0]
    

    ## filename will act as a unique URI to connect all the three graphs
    filesubject = dcc_namespace + filename
    # consolidatedGraph
    consolidatedGraph.add((URIRef(filesubject),RDF.type,URIRef(dcc_namespace + "Publication")))
    triple_list.append(filename + " isa " + "Publication")
    year = Literal(row['year'])
    conference = Literal(row['conference'])
    platform = Literal(row['Platform'])

    consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "yearOfPublication"),year ))
    consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "conferenceSeries"),conference ))
    consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "platform"),platform ))

    # Just the triple list
    triple_list.append(filename + " year_of_publication " + str(row['year']))
    triple_list.append(filename + " conference_series " + str(row['conference']))
    triple_list.append(filename + " platform " + str(row['Platform']))

    textfilename = text_dir + filename + ".txt"
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
            # print("CSO label found for entity text : " + entitytext  + " : and value is " + entity_map[entitytext])
        # print(entitytext)
        # print(filesubject + "_" + entitytext)
        consolidatedGraph.add((URIRef(filesubject + "_" + entitytext),RDF.type,URIRef(dcc_namespace + entitylabel)))
        consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "hasEntity"),URIRef(filesubject + "_" + entitytext)))
        textLiteral = Literal(entitytext)
        consolidatedGraph.add((URIRef(filesubject + "_" + entitytext),URIRef(dcc_namespace + 'hasText'),textLiteral))

        triple_list.append(entitytext + " isa " + entitylabel)
        # triple_list.append(filename + " has entity " + )


    print("Done with file " + filename)
    return(filename, triple_list)
    


if __name__ == '__main__':

    # Load paths
    config = yaml.safe_load(open('../../conf/conf.yaml'))
    model_dir=config['MODEL_PATH']
    ontology=config['ONTOLOGY_PATH']
    destinationfolder = config['EXTRACT_TEXT_RDF_GRAPH_PATH']
    triple_dir = config['EXTRACT_TEXT_TRIPLE_GRAPH_PATH']
    csvfile = config['TEXT_GRAPH_CSV']
    text_dir = config['TEXT_GRAPH_PAPERS']


    # load ontology:
    consolidatedGraph = Graph() 
    consolidatedGraph.parse(ontology,format="n3")

    #text2graphs are present in the text2graph.csv
    df = pd.read_csv(csvfile)
    # df.head()


    #load CSO
    f = open(os.path.join(model_dir,'full_annotations.pcl'), 'rb')
    [entity_map,uri2entity, uri2rel]=pickle.load(f)
    f.close()
    # print(entity_map.items())


    # For each paper from text2graph.csv create triples (embedding) and text2graph(rdf)
    for index,row in df.iterrows():
        filename, triple_list= createrdf(row,entity_map,consolidatedGraph)
        save_triple_file(triple_list,filename)
                                        
    # print("Total files converted now are " + filecount)
    print("Saving final consolidated rdf file ")
    destinationfile = destinationfolder + "text2graph_v4.ttl"
    print("Saving final consolidated rdf file : " + destinationfile)
    consolidatedGraph.serialize(destination=destinationfile,format='turtle')
    
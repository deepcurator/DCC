from __future__ import unicode_literals, print_function
import os
import os
from pathlib import Path
import glob
import pandas as pd

# RDF specific libraries

from rdflib import URIRef, BNode, Literal
from rdflib.namespace import RDF, FOAF 
import rdflib
from rdflib import Graph
import pprint

# spacy specific libraries
import spacy
import plac

#nltk libraries
import nltk


# Ontology locations
ontology = "/home/z003z47y/git/DCC/src/ontology/DeepSciKG.nt"
destinationfolder = "/home/z003z47y/git/DCC/src/text2graph/Output/rdf/"

# csv file location
csvfile = '/home/z003z47y/git/DCC/src/text2graph.csv'


filecount = 0
noimage2graphlist = []
model_dir = './Models/'
text_dir = '/home/z003z47y/git/DCC/src/text2graph/NewPapers/'
image_dir = '/home/z003z47y/Projects/Government/ASKE/2019/092219_output/'
image_dir_addon = 'diag2graph'
output_dir = './Output/'
rdf_dir = 'Output/rdf/'


def getabstract(filename):
    abstxt = ""
    with open(filename) as file:
        lines = file.readlines()
        abstxt = ''.join(str(line) for line in lines)
        abstxt = abstxt.lower()
    return abstxt

consolidatedGraph = Graph() 
consolidatedGraph.parse(ontology,format="n3")


def createrdf(row):
    # image2graphfiles = []
    g = Graph()
    g.parse(ontology,format="n3")
    dcc_namespace = "https://github.com/deepcurator/DCC/"

    # print(row['paper_title'],row['paper_link'],row['conference'], row['year'], row['Platform'])
    filename = row['paper_link'].split('/')[-1]
    if(filename.endswith('.pdf')):
        filename = filename.split('.pdf')[0]
    elif(filename.endswith('.html')):
        filename = filename.split('.html')[0]
    
    # print(filename)
    textfilename = text_dir + filename + ".txt"
    # print(textfilename)
    # print(getabstract(textfilename))

    ## filename will act as a unique URI to connect all the three graphs

    filesubject = dcc_namespace + filename
    # g.add((URIRef(filesubject),RDF.type,URIRef(dcc_namespace + "Publication")))
    # consolidatedGraph
    consolidatedGraph.add((URIRef(filesubject),RDF.type,URIRef(dcc_namespace + "Publication")))
    year = Literal(row['year'])
    conference = Literal(row['conference'])
    platform = Literal(row['Platform'])

    consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "yearOfPublication"),year ))
    consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "conferenceSeries"),conference ))
    consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "platform"),platform ))
   
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
    # print("---------------------------------")

    # image2graphfiles = glob.glob(image_dir + filename + "/" + image_dir_addon + "/*.txt")

    # file_length = len(image2graphfiles)
    # if(file_length > 0):
    #     for file in image2graphfiles:
    #         createimage2graph(file,ontology,filesubject)
    # else :
    #     noimage2graphlist.append(filename)

    # print(len(image2graphfiles))
    # print(image2graphfiles)
    print("Done with file " + filename)
    

try:
	os.chdir(os.path.join(os.getcwd(), 'src/text2graph'))
	print(os.getcwd())
except:
	pass

df = pd.read_csv(csvfile)
df.head()



# iterate through the rows in the dataframe

for index,row in df.iterrows():
    createrdf(row)



# print("Total files converted now are " + filecount)
print("Saving final consolidated rdf file ")
destinationfile = destinationfolder + "text2graph.ttl"
print("Saving final consolidated rdf file : " + destinationfile)
consolidatedGraph.serialize(destination=destinationfile,format='turtle')
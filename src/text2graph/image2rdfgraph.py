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

consolidatedGraph = Graph() 
consolidatedGraph.parse(ontology,format="n3")

image2graphs = []

for root, dirs, files in os.walk("/home/z003z47y/Projects/Government/ASKE/2019/092219_output/"):
    for file in files:
        if file.endswith('.txt'):
            # print(os.path.join(root, file))
            image2graphs.append(os.path.join(root, file))



def createimage2graph(inputfile,ontology,filesubject):
    
    #filesubject is the publication URI, which has to be linked to the image components

    # g = Graph()
    # g.parse(ontology,format="n3")
    # len(g)

    block_dict = {
        "Figure":"Figure",
        "conv": "ConvBlock",
        "deconv":"DeconvBlock",
        "dense":"DenseBlock",
        "flatten":"FlattenBlock",
        "dropout":"DropoutBlock",
        "pooling":"PoolingBlock",
        "unpooling":"UnpoolingBlock",
        "concat":"ConcatBlock",
        "rnn":"RnnBlock",
        "rnnseq": "RnnSeqBlock",
        "lstm":"LSTMBlock",
        "lstmseq":"LSTMSeqBlock",
        "norm":"NormBlock",
        "embed":"EmbedBlock",
        "activation":"ActivationBlock",
        "loss":"LossBlock",
        "output":"OutputBlock",
        "input":"InputBlock"
    }

# Namespaces
    dcc_namespace = "https://github.com/deepcurator/DCC/"

    # Classes
    Figure = URIRef(dcc_namespace + "Figure")
    # ActivationBlock = URIRef(dcc_namespace + "ActivationBlock")
    # EmbedBlock = URIRef(dcc_namespace + "EmbedBlock")
    # NormBlock = URIRef(dcc_namespace + "NormBlock")
    # LSTMSeqBlock = URIRef(dcc_namespace + "LSTMSeqBlock")
    # LSTMBlock = URIRef(dcc_namespace + "LSTMBlock")
    # RNNSeqBlock = URIRef(dcc_namespace + "RNNSeqBlock")
    # RNNBlock = URIRef(dcc_namespace + "RNNBlock")
    # ConcatBlock = URIRef(dcc_namespace + "ConcatBlock")
    # UnpoolingBlock = URIRef(dcc_namespace + "UnpoolingBlock")
    # PoolingBlock = URIRef(dcc_namespace + "PoolingBlock")
    # DropoutBlock = URIRef(dcc_namespace + "DropoutBlock")
    # FlattenBlock = URIRef(dcc_namespace + "FlattenBlock")
    # DenseBlock = URIRef(dcc_namespace + "DenseBlock")
    # DeconvBlock = URIRef(dcc_namespace + "DeconvBlock")
    # ConvBlock = URIRef(dcc_namespace + "ConvBlock")
    # LossBlock = URIRef(dcc_namespace + "LossBlock")
    # Properties
    partOf = URIRef(dcc_namespace + "partOf")
    followedBy = URIRef(dcc_namespace + "followedBy")

    # Open the image2graph

    with open(inputfile,encoding="ISO-8859-1") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]

    # Each line in the image2graph is a triple
    # Split the triple into s,p,o
    # Create the URIRefs for RDF based on the ontology
    # URIRefs require the namespace and the class term from ontology

    for line in lines:
        triple = line.split(" ")
        subject = triple[0]
        predicate = triple[1]
        obj = triple[2]

        filename = inputfile.split('/')[-1]
        filename = filename.split('.txt')[0]
        

        if (subject.startswith(":")):
            subject = subject[1:]
        if (obj.startswith(":")):
            obj = obj[1:]
        # print(line + "\n")
        if(predicate == "partOf"):
            ## Subject is a component
            ## Create a unique URI for that
            filename = inputfile.split('/')[-1]
            filename = filename.split('.txt')[0]
            subject = URIRef(dcc_namespace + filename[4:] + "_" + subject[4:])
            obj = URIRef(dcc_namespace + obj[4:])
            # g.add((subject,partOf,obj))
            consolidatedGraph.add((subject,partOf,obj))
        elif(predicate == "isA"):
            subject = URIRef(dcc_namespace + subject[4:])
            # g.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            if(obj == "Figure"):
                consolidatedGraph.add((URIRef(dcc_namespace + filesubject),URIRef(dcc_namespace + "hasFigure"),subject))
            consolidatedGraph.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
        elif(predicate == "isType"):
            filename = inputfile.split('/')[-1]
            filename = filename.split('.txt')[0]
            subject = URIRef(dcc_namespace + filename[4:] + "_" + subject)
            # g.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            consolidatedGraph.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))

for image2graph in image2graphs:
    print("Working on file " + image2graph)
    filesubject = image2graph.split('/')[-3]
    print(filesubject)
    createimage2graph(image2graph,ontology,filesubject)


print("Saving final consolidated rdf file ")
destinationfile = destinationfolder + "image2graph.ttl"
print("Saving final consolidated rdf file : " + destinationfile)
consolidatedGraph.serialize(destination=destinationfile,format='turtle')

print(len(image2graphs))
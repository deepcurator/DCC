from __future__ import unicode_literals, print_function
import os
from os import listdir
from os.path import isfile, join
# from pathlib import Path
from glob import glob
import pandas as pd

# RDF specific libraries
from rdflib import URIRef, BNode, Literal
from rdflib.namespace import RDF, FOAF 
import rdflib
from rdflib import Graph
import pprint

def createimage2graph(inputfile, entity_map, ontology, filesubject, g):
    
    #filesubject is the publication URI, which has to be linked to the image components

    # g = Graph()
    # g.parse(ontology,format="n3")
    # len(g)

    inputfile_uri = inputfile.replace("\\", "/")
    
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

    # Properties
    partOf = URIRef(dcc_namespace + "partOf")
    followedBy = URIRef(dcc_namespace + "followedBy")

    # Open the image2graph

    with open(inputfile, encoding="ISO-8859-1") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]

    # Each line in the image2graph is a triple
    # Split the triple into s,p,o
    # Create the URIRefs for RDF based on the ontology
    # URIRefs require the namespace and the class term from ontology

    for line in lines:
        triple = line.split(" ")
        # print(triple)
        if (len(triple) != 3):
            continue
        subject = triple[0]
        predicate = triple[1]
        obj = triple[2]
        
        if (subject.startswith(":")):
            subject = subject[1:]
        #if (predicate.startswith(":")):
            #predicate = predicate[1:]
        if (obj.startswith(":")):
            obj = obj[1:]
        
        # print(line + "\n")
        if(predicate == "partOf"):
            ## Subject is a component
            ## Create a unique URI for that
            filename_list = inputfile_uri.split('/')
            if (len(filename_list) == 0):
                filename_list = inputfile_uri.split('\\')
            filename = filename_list[-1]
            
            # print(filename)
            filename = filename.split('.txt')[0]
            # print("------ " + filename)
            # subject = URIRef(dcc_namespace + filename[4:] + "_" + subject[1:])
            subject = URIRef(dcc_namespace + filename + "_" + subject)
            obj = URIRef(dcc_namespace + obj)
            # g.add((subject,partOf,obj))
            g.add((subject,partOf,obj))
        elif(predicate == "isA"):
            subject = URIRef(dcc_namespace + subject)
            # g.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            if(obj == "Figure"):
                g.add((URIRef(filesubject),URIRef(dcc_namespace + "hasFigure"),subject))
            g.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
        elif(predicate == "isType"):
            flist = inputfile_uri.split('/')
            if (len(flist) < 2):
                flist = inputfile_uri.split('\\')
            filename = flist[-1]
            filename = filename.split('.txt')[0]
            # subject = URIRef(dcc_namespace + filename[4:] + "_" + subject[1:])
            subject = URIRef(dcc_namespace + filename + "_" + subject)
            # g.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            g.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
        
        
        # Link CSO 
        if(obj in entity_map):
            # print("found obj in entity map")
            # print("Found " + obj + " in cso")
            csovalue = entity_map[obj]
            str_value = str(csovalue)
            # print("CSO value is then " + str_value)
            if("cso" in str_value):
                g.add((subject,URIRef(dcc_namespace + "hasCSOEquivalent"),csovalue))

    # All triples are created for the current file
    # Serialize the rdf files to their right folder

    # filename = inputfile.split('/')[-1]
    # filename = filename.split('.txt')[0]
    # print(destinationfolder + filename[4:])
    # destinationfile = destinationfolder + filename[4:] + ".ttl"
    # print("Saving rdf graph " + destinationfile + "\n")
    # g.serialize(destination=destinationfile,format='turtle')

image_dir_addon = 'diag2graph'

def runI2G(paper_dir, entity_map, image_triple_dir, image_output_dir, ontology_file):
    dcc_namespace = "https://github.com/deepcurator/DCC/"
    g = Graph()
    # ontology = "C:/dcc_test/demo/DeepSciKG.nt"
    
    # g.parse(ontology_file, format="n3") 
    
    paperList = [f for f in listdir(paper_dir) if isfile(join(paper_dir, f))]
    # onlyFiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_triple_dir) for f in filenames if os.path.splitext(f)[1] == '.txt']
    # onlyFiles = glob(image_triple_dir + '/**/*.txt', recursive=True)
    # print(onlyFiles)
    for paper in paperList:
        paper_name = paper.replace("\\", "/")
        print("Processing paper: " + paper_name)
        if (paper_name.endswith(".pdf")):
            # print(os.path.splitext(file))
            # paper_name_short_start = paper_name.rfind('/') + 1
            # paper_name_short = paper_name[paper_name_short_start:]
            paper_name_short = paper_name.replace(".pdf", "")
            filesubject = dcc_namespace + paper_name_short
            print(image_triple_dir + "/" + paper_name_short + "/" + image_dir_addon)
            image2graphfiles = glob(image_triple_dir + "/" + paper_name_short + "/" + image_dir_addon + "/*.txt")
            print(image2graphfiles)
            for image2graphfile in image2graphfiles:
                createimage2graph(image2graphfile, entity_map, None, filesubject, g)
        
        destination_dir = image_triple_dir + "/" + paper_name_short
        destinationFile = image_triple_dir + "/" + paper_name_short + "/image2graph.ttl"
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        g.serialize(destination=destinationFile, format='turtle')

# image_triple_dir = "C:/aske-2/dcc/grobid-workspace/image_input/diag2graph/"
# image_output_dir = "C:/aske-2/dcc/grobid-workspace/image_output/"

# runI2G(image_triple_dir, image_output_dir)
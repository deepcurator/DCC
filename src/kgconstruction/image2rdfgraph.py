from __future__ import unicode_literals, print_function
import os
import os
from pathlib import Path
import glob
import pandas as pd
import yaml
import pickle
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




def save_image_triple_file(triple_list,topdirectory,bottomdirectory, triple_dir):
    # print("Entering save image triple file")
    # triplefilename = triple_dir + topdirectory + "/" + bottomdirectory + "/" + "i2g.triples"
    if not os.path.exists(os.path.join(triple_dir,topdirectory)):
        os.mkdir(os.path.join(triple_dir,topdirectory))
    # if not os.path.exists(os.path.join(triple_dir,topdirectory,bottomdirectory)):
    #     os.mkdir(os.path.join(triple_dir,topdirectory,bottomdirectory))
    triplefilename = os.path.join(triple_dir,topdirectory,bottomdirectory+".triples")
    with open(triplefilename,'w') as f :
        for line in triple_list:
            f.write(line + "\n")
    # print("image triple file saved at " + triplefilename)
def createimage2graph(inputfile, entity_map, ontology, filesubject, lowerlevel, consolidatedGraph, triple_dir, generateEmbTriples):
    
    #filesubject is the publication URI, which has to be linked to the image components

    # g = Graph()
    # g.parse(ontology,format="n3")
    # len(g)
    imagetriples = []

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
        "input":"InputBlock",
        "upsample":"UpsamplingBlock"
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

        filename = inputfile.split(os.path.sep)[-1]
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
            
            # print(subject + "\tpart of\t" + obj[4:])
            imagetriples.append(subject.replace("\\", "/") + "\tpart of\t" + obj[4:].replace("\\", "/"))
            subject = URIRef(dcc_namespace + filename[4:].replace("\\", "/") + "_" + subject.replace("\\", "/"))
            obj = URIRef(dcc_namespace + obj[4:].replace("\\", "/"))
            # g.add((subject,partOf,obj))
            
            consolidatedGraph.add((subject,partOf,obj))
        elif(predicate == "hasCaption"):
            triplesubj = subject 
            subject = URIRef(dcc_namespace + subject)
            literaltext = Literal(obj)
            consolidatedGraph.add((subject,URIRef(dcc_namespace + "hasCaptionText"),literaltext))

        elif(predicate == "isA"):
            triplesubj = subject 
            subject = URIRef(dcc_namespace + subject)
            
            # if(obj in entity_map):
            #     print("found obj in entity map")
            #     print("Found " + obj + " in cso")
            #     csovalue = entity_map[obj]
            #     str_value = str(csovalue)
            #     print("CSO value is then " + str_value)

            
            # g.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            if(obj == "Figure"):
                # print(filesubject + "\thas Figure\t" + obj)
                imagetriples.append(filesubject + "\thas Figure\t" + obj)
                consolidatedGraph.add((URIRef(dcc_namespace + filesubject),URIRef(dcc_namespace + "hasFigure"),subject))
            
            # print(triplesubj +"\tisA\t" + block_dict.get(obj))

            imagetriples.append(triplesubj +"\tisA\t" + block_dict.get(obj))
            consolidatedGraph.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
        elif(predicate == "isType"):

            filename = inputfile.split(os.path.sep)[-1]
            # print("FILENAME: " + filename)
            filename = filename.split('.txt')[0]
            # print(subject + "\tisA\t" + block_dict.get(obj))
            # print(obj)
            imagetriples.append(subject + "\tisA\t" + block_dict.get(obj))
            subject = URIRef(dcc_namespace + filename[4:] + "_" + subject)
            # print("Subject is " + subject)
            # g.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            consolidatedGraph.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))

            # Link CSO 
            if(obj in entity_map):
                # print("found obj in entity map")
                # print("Found " + obj + " in cso")
                csovalue = entity_map[obj]
                str_value = str(csovalue)
                # print("CSO value is then " + str_value)
                if("cso" in str_value):
                    consolidatedGraph.add((subject,URIRef(dcc_namespace + "hasCSOEquivalent"),csovalue))


    if generateEmbTriples:
        save_image_triple_file(imagetriples,filesubject,lowerlevel, triple_dir)


def run_batch():
    config = yaml.safe_load(open('../../conf/conf.yaml'))

    # Ontology location
    ontology = config["ONTOLOGY_PATH"]
    #RDF file save location
    destinationfolder = config["EXTRACT_TEXT_RDF_GRAPH_PATH"]
    #Image triples save location
    triple_dir = config["EXTRACT_IMAGE_TRIPLE_GRAPH_PATH"]
    #image to graph pipeline output location
    input_triple_dir = config["IMAGE_TO_GRAPH_OUTPUT_TEXT"]

    consolidatedGraph = Graph() 
    consolidatedGraph.parse(ontology,format="n3")

    image2graphs = []

    for root, dirs, files in os.walk(input_triple_dir):
        for file in files:
            if file.endswith('.txt'):
                # print(os.path.join(root, file))
                image2graphs.append(os.path.join(root, file))

    model_dir = config['MODEL_PATH']
    # print(model_dir)


    f = open(os.path.join(model_dir, 'full_annotations.pcl'), 'rb')
    [entity_map,uri2entity, uri2rel]=pickle.load(f)
    f.close()

    for image2graph in image2graphs:
        print("Working on file " + image2graph)
        filesubject = image2graph.split('/')[-3]

        # print("Top folder is " + image2graph.split('/')[-3])
        lowerlevel = image2graph.split('/')[-1]
        lowerlevel = lowerlevel[4:]
        filenameLength = len(filesubject)+1
        lowerlevel = lowerlevel[filenameLength:]
        # print("image2graph is " + image2graph)
        # print("filesubject is " + filesubject)
        # print("Length of filename is " + str(filenameLength))
        # print("Image2graph split is " + image2graph.split('/')[-1][4:][filenameLength:])
        # print("Lower Level is " + lowerlevel)
        # print("Lower Level split is " + lowerlevel.split('.')[-2])
        # # print(filesubject)
        # createimage2graph(inputfile, entity_map, ontology, filesubject, lowerlevel, consolidatedGraph, triple_dir, generateEmbTriples):
        createimage2graph(image2graph,entity_map,ontology,filesubject,lowerlevel.split('.')[-2], consolidatedGraph, triple_dir, True)


    print("Saving final consolidated rdf file ")
    destinationfile = destinationfolder + "image2graph.ttl"
    print("Saving final consolidated rdf file : " + destinationfile)
    consolidatedGraph.serialize(destination=destinationfile,format='nt')

    print("Total of " + str(len(image2graphs)) + " files were converted to image to graphs ( RDF and triples)")


if __name__ == '__main__':
    run_batch()
from rdflib import URIRef, BNode, Literal
from rdflib.namespace import RDF, FOAF 
import rdflib
from rdflib import Graph
import pprint

import os,glob
# os.chdir("/home/z003z47y/Projects/Government/ASKE/2019/092219_output/")
# for file in glob.glob("*.txt"):
#     print(file)
image2graphs = []
outputList = set()
ontology = "/home/z003z47y/git/DCC/src/ontology/DeepSciKG.nt"
destinationfolder = "/home/z003z47y/git/DCC/src/ontology/rdf/image2graph/"
for root, dirs, files in os.walk("/home/z003z47y/Projects/Government/ASKE/2019/092219_output/"):
    for file in files:
        if file.endswith('.txt'):
            # print(os.path.join(root, file))
            image2graphs.append(os.path.join(root, file))


consolidatedGraph = Graph() 
consolidatedGraph.parse(ontology,format="n3")

def createPredicateSet(inputfile):
    with open(inputfile,encoding="ISO-8859-1") as f:
        lines = f.readlines()
    predicateLines = [x.strip() for x in lines]
    for line in predicateLines:
        triple = line.split(" ")
        subject = triple[0]
        predicate = triple[1]
        if(triple[2] is None):
            obj = ""
        else:
            obj = triple[2]
        # obj = triple[2]
        # print("obj is " + obj)
        if(predicate=="isType"):
            outputList.add(obj)
    print(outputList)


    
# for file in image2graphs:
#     print(file + "\n")
#     createPredicateSet(file)
# print(outputList)
# print(len(outputList))
def createimage2graph(inputfile,ontology,destinationfolder):

    g = Graph()
    g.parse(ontology,format="n3")
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
        # print(line + "\n")
        if(predicate == "partOf"):
            ## Subject is a component
            ## Create a unique URI for that
            filename = inputfile.split('/')[-1]
            filename = filename.split('.txt')[0]
            subject = URIRef(dcc_namespace + filename[4:] + "_" + subject[1:])
            obj = URIRef(dcc_namespace + obj)
            g.add((subject,partOf,obj))
            consolidatedGraph.add((subject,partOf,obj))
        elif(predicate == "isA"):
            subject = URIRef(dcc_namespace + subject)
            g.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            consolidatedGraph.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
        elif(predicate == "isType"):
            filename = inputfile.split('/')[-1]
            filename = filename.split('.txt')[0]
            subject = URIRef(dcc_namespace + filename[4:] + "_" + subject[1:])
            g.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            consolidatedGraph.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))

    # All triples are created for the current file
    # Serialize the rdf files to their right folder

    filename = inputfile.split('/')[-1]
    filename = filename.split('.txt')[0]
    print(destinationfolder + filename[4:])
    destinationfile = destinationfolder + filename[4:] + ".ttl"
    print("Saving rdf graph " + destinationfile + "\n")
    g.serialize(destination=destinationfile,format='turtle')

# print(image2graphs)
for file in image2graphs:
    filename = file.split('/')[-1]
    filename = filename.split('.txt')[0]
    print("Creating RDF graph for " + filename[4:])
    createimage2graph(file,ontology,destinationfolder)

consolidatedGraph.serialize(destination="image2graph.ttl",format='turtle')
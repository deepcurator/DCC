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
csvfile = '/home/z003z47y/git/DCC/src/pwc_graphs_current.csv'


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
        # print(line + "\n")
        if(predicate == "partOf"):
            ## Subject is a component
            ## Create a unique URI for that
            filename = inputfile.split('/')[-1]
            filename = filename.split('.txt')[0]
            subject = URIRef(dcc_namespace + filename[4:] + "_" + subject[1:])
            obj = URIRef(dcc_namespace + obj)
            # g.add((subject,partOf,obj))
            consolidatedGraph.add((subject,partOf,obj))
        elif(predicate == "isA"):
            subject = URIRef(dcc_namespace + subject)
            # g.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            if(obj == "Figure"):
                consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "hasModality"),subject))
            consolidatedGraph.add((subject,RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
        elif(predicate == "isType"):
            filename = inputfile.split('/')[-1]
            filename = filename.split('.txt')[0]
            subject = URIRef(dcc_namespace + filename[4:] + "_" + subject[1:])
            # g.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))
            consolidatedGraph.add((subject, RDF.type, URIRef(dcc_namespace + block_dict.get(obj))))

    # All triples are created for the current file
    # Serialize the rdf files to their right folder

    # filename = inputfile.split('/')[-1]
    # filename = filename.split('.txt')[0]
    # print(destinationfolder + filename[4:])
    # destinationfile = destinationfolder + filename[4:] + ".ttl"
    # print("Saving rdf graph " + destinationfile + "\n")
    # g.serialize(destination=destinationfile,format='turtle')

def createrdf(row):
    image2graphfiles = []
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

    image2graphfiles = glob.glob(image_dir + filename + "/" + image_dir_addon + "/*.txt")

    file_length = len(image2graphfiles)
    if(file_length > 0):
        for file in image2graphfiles:
            createimage2graph(file,ontology,filesubject)
    else :
        noimage2graphlist.append(filename)

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


print("Files not having image2graphs : \n" )
print(noimage2graphlist)
print("------------------------------------------")
# print("Total files converted now are " + filecount)
print("Saving final consolidated rdf file ")
destinationfile = destinationfolder + "consolidated.ttl"
print("Saving final consolidated rdf file : " + destinationfile)
consolidatedGraph.serialize(destination=destinationfile,format='turtle')
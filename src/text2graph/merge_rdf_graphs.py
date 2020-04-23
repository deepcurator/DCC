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


# Load paths
config = yaml.safe_load(open('../../conf/conf.yaml'))
model_dir=config['MODEL_PATH']
ontology=config['ONTOLOGY_PATH']
destinationfolder = config['EXTRACT_TEXT_RDF_GRAPH_PATH']
triple_dir = config['EXTRACT_TEXT_TRIPLE_GRAPH_PATH']
csvfile = config['TEXT_GRAPH_CSV']
text_dir = config['TEXT_GRAPH_PAPERS']

mergedGraph = Graph()
mergedGraph.parse(destinationfolder + 'image2graph.ttl',format='ttl')
mergedGraph.parse(destinationfolder + 'text2graph.ttl',format='ttl')
# mergedGraph.parse(destinationfolder + 'code2graph.ttl',format='ttl')
    
mergedGraph.serialize(destination=destinationfolder + 'consolidated_image_text_4_22_2020.ttl',format='nt')
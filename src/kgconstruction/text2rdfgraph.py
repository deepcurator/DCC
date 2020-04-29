from __future__ import unicode_literals, print_function
import os
import sys
from pathlib import Path
import glob
import pandas as pd
import yaml
import pickle
import subprocess
from os import listdir
from os.path import isfile, join
sys.path.append(os.path.abspath('../../src'))
from text2graph.xml2txt_no_sents import TEIFile

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

def createrdf(filepath, text_dir, year, conference, platform, entity_map, consolidatedGraph):

    config = yaml.safe_load(open('../../conf/conf.yaml'))
    model_dir = config['MODEL_PATH']

    triple_list = []

    dcc_namespace = "https://github.com/deepcurator/DCC/"

    # print(row['paper_title'],row['paper_link'],row['conference'], row['year'], row['Platform'])
    filename = filepath.split('/')[-1]
    if(filename.endswith('.pdf')):
        filename = filename.split('.pdf')[0]
    elif(filename.endswith('.html')):
        filename = filename.split('.html')[0]
    

    ## filename will act as a unique URI to connect all the three graphs
    filesubject = dcc_namespace + filename
    # consolidatedGraph
    consolidatedGraph.add((URIRef(filesubject),RDF.type,URIRef(dcc_namespace + "Publication")))
    triple_list.append(filename + " isa " + "Publication")
    year = Literal(year)
    conference = Literal(conference)
    platform = Literal(platform)

    consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "yearOfPublication"),year ))
    consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "conferenceSeries"),conference ))
    consolidatedGraph.add((URIRef(filesubject),URIRef(dcc_namespace + "platform"),platform ))

    # Just the triple list
    triple_list.append(filename + " year_of_publication " + str(year))
    triple_list.append(filename + " conference_series " + str(conference))
    triple_list.append(filename + " platform " + str(platform))

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
    
def run_demo(input_dir, output_dir, ontology_file, model_dir, grobid_client):
    # output_dir = "C:/aske-2/dcc/grobid-workspace/output"
    
    # ontology = "C:/dcc_test/demo/DeepSciKG.nt"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    t2g_output_dir = output_dir + "/text2graph"
    
    if not os.path.exists(t2g_output_dir):
        os.makedirs(t2g_output_dir)
    
    # command = "python grobid-client-python/grobid-client.py --config grobid-client-python/config.json --input C:/aske-2/dcc/grobid-workspace/input --output C:/aske-2/dcc/grobid-workspace/output processFulltextDocument"
    # command = "python grobid-client-python/grobid-client.py --config grobid-client-python/config.json --input " + input_dir + " --output " + t2g_output_dir + " processFulltextDocument"
    command = "python " + grobid_client + "/grobid-client.py --config grobid-client-python/config.json --input " + input_dir + " --output " + t2g_output_dir + " processFulltextDocument"

    #process = Popen(command, shell=True)
    #stdout, stderr = process.communicate()
    
    print("[Info] Extracting XML from PDF's...")
    
    # subprocess.call(["python", "grobid-client-python\grobid-client.py", "--config", "grobid-client-python\config.json", "--input", "C:\aske-2\dcc\grobid-workspace\input", "-–output", "C:\aske-2\dcc\grobid-workspace\output", "C:\aske-2\dcc\grobid-workspace\output", "processFulltextDocument"])
    # , stdout=subprocess.PIPE, stderr=subprocess.PIPE
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #process.wait()
    stdout, stderr = process.communicate()
    # print(stdout)
    # print(stderr)
    
    #process.wait()
    
    ################## pip install lxml 
    
    print("[Info] Extracting abstracts from XML's...")
    
    # mypath = "C:/aske-2/dcc/grobid-workspace/output"
    onlyXMLfiles = [f for f in listdir(t2g_output_dir) if isfile(join(t2g_output_dir, f))]
    
    for i, f in enumerate(onlyXMLfiles):
        if f.endswith("xml"):
            # print('Processing paper ', i)
            # tei_file = join(t2g_output_dir, f)
            tei_file = t2g_output_dir + "/" + f
            paper = TEIFile(tei_file)
        
        
            paper_body = paper.partial_text
            paper_body = paper_body.lower()
            paper_body = paper_body.replace('et al.', 'et al')
            new_f = f.replace('.tei', '')
            new_f = new_f.replace('.xml', '')
        
        
            outfile = join(t2g_output_dir, new_f + '.txt')
            with open(outfile, 'w', encoding='utf8') as of:
                of.write(paper_body)
    
    print("[Info] Extracting entities/relationships and generating RDF's...")
    
    # inputFolder = 'C:/aske-2/dcc/grobid-workspace/output/'
    # outputFolder = 'C:/aske-2/dcc/grobid-workspace/output/'
    # rdf_folder = mypath + "/"
    rdf_input_dir = t2g_output_dir + "/"
    rdf_output_dir = t2g_output_dir + "/"
    # createTextRDF(rdf_input_dir, rdf_output_dir, ontology_file, model_dir)
    
    onlyFiles = [f for f in listdir(rdf_input_dir) if isfile(join(rdf_input_dir, f))]
    
    #load CSO
    f = open(os.path.join(model_dir,'full_annotations.pcl'), 'rb')
    [entity_map,uri2entity, uri2rel]=pickle.load(f)
    f.close()
    
    # iterate through the rows in the dataframe
    #  for index,row in df.iterrows():
    for f in onlyFiles:
        g = Graph() 
        # g.parse(ontology_file, format="n3") 
        if f.endswith(".txt"):
            createrdf(f.replace(".txt", ""), rdf_output_dir, "", "", "", entity_map, g)
            destinationfile = rdf_output_dir + f[:-4] + "_text2graph.ttl"
            print("Saving rdf file " + destinationfile)
            g.serialize(destination=destinationfile, format='turtle')
    
    print("[Info] Completed text2graph pipeline!")

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
        paper_link = row['paper_link']
        year = row['year']
        conference = row['conference']  
        plaform = row['Platform']
        filename, triple_list= createrdf(paper_link, text_dir, year, conference, plaform, entity_map, consolidatedGraph)
        save_triple_file(triple_list,filename)
                                        
    # print("Total files converted now are " + filecount)
    print("Saving final consolidated rdf file ")
    destinationfile = destinationfolder + "text2graph.ttl"
    print("Saving final consolidated rdf file : " + destinationfile)
    consolidatedGraph.serialize(destination=destinationfile,format='nt')
    
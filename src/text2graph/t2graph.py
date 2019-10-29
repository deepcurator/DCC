import subprocess
# import sys
# sys.path.insert(0, "/grobid-client-python")
import os
from os import listdir
from os.path import isfile, join
from .xml2txt_no_sents import TEIFile
from .text_rdf_generator import createTextRDF

def run(input_dir, output_dir, ontology_file, model_dir):
    # output_dir = "C:/aske-2/dcc/grobid-workspace/output"
    
    # ontology = "C:/dcc_test/demo/DeepSciKG.nt"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # command = "python grobid-client-python/grobid-client.py --config grobid-client-python/config.json --input C:/aske-2/dcc/grobid-workspace/input --output C:/aske-2/dcc/grobid-workspace/output processFulltextDocument"
    command = "python grobid-client-python/grobid-client.py --config grobid-client-python/config.json --input " + input_dir + " --output " + output_dir + " processFulltextDocument"

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
    onlyXMLfiles = [f for f in listdir(output_dir) if isfile(join(output_dir, f))]
    
    for i, f in enumerate(onlyXMLfiles):
        if f.endswith("xml"):
            # print('Processing paper ', i)
            # tei_file = join(output_dir, f)
            tei_file = output_dir + "/" + f
            paper = TEIFile(tei_file)
        
        
            paper_body = paper.partial_text
            paper_body = paper_body.lower()
            paper_body = paper_body.replace('et al.', 'et al')
            new_f = f.replace('.tei', '')
            new_f = new_f.replace('.xml', '')
        
        
            outfile = join(output_dir, new_f + '.txt')
            with open(outfile, 'w', encoding='utf8') as of:
                of.write(paper_body)
    
    print("[Info] Extracting entities/relationships and generating RDF's...")
    
    # inputFolder = 'C:/aske-2/dcc/grobid-workspace/output/'
    # outputFolder = 'C:/aske-2/dcc/grobid-workspace/output/'
    # rdf_folder = mypath + "/"
    rdf_input_dir = output_dir + "/"
    rdf_output_dir = output_dir + "/"
    createTextRDF(rdf_input_dir, rdf_output_dir, ontology_file, model_dir)
    
    print("[Info] Completed text2graph pipeline!")
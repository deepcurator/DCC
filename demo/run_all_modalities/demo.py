import sys
import os
import warnings

# Disable printing - write into the log file
def disablePrint():
    sys.stdout = open("log.txt", 'w')

disablePrint()

# Ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

sys.path.append(os.path.abspath('../../src'))

import text2graph
import diagram2graph
import code2graph
import pytesseract
import IPython
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from visualize import get_vis
from rdflib import Graph, URIRef

from text2graph import t2graph
from diagram2graph.FigAnalysis.ShapeExtraction import i2graph
from code2graph import c2graph

import urllib
import git
import shutil
import yaml

from rdflib import Graph, URIRef
from visualize import get_vis

# %load_ext autoreload
# %autoreload 2



print("Necessary modules have been successfully imported!")

# Read YAML file
with open("conf.yaml", 'r') as stream:
    conf = yaml.safe_load(stream)

ontology_file = conf["ontology_file"]
text2graph_models_dir = conf["text2graph_models_dir"]
image2graph_models_dir = conf["image2graph_models_dir"]
grobid_client = conf["grobid_client"]
pytesseract.pytesseract.tesseract_cmd = conf["tesseract_cmd"]

# --------- INPUT ---------
# Input paper pdf will be downloaded into this folder
inputFolder = 'demo_input_demo2'

# Input paper code will be downloaded into this folder
codeFolder = "demo_code_demo2"

# CSV file that maps the pdf file name in the inputFolder to the code repository name in the codeFolder
# This file will be generated on the fly
inputCSV = 'input_demo2.csv'

# --------- OUTPUT ---------
# Output will be placed into this folder
outputFolder = 'demo_output_demo2'




# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    # sys.stdout = stdout
    

def download_file(download_url):
    response = urllib.request.urlopen(download_url)
    file = open(inputFolder + "/input_paper.pdf", 'wb')
    file.write(response.read())
    file.close()
    print("Completed")

def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

def visualize_graphs(rels):
    g = Graph()
    g.parse(outputFolder + "/text2graph/input_paper_text2graph.ttl", format="ttl")
    g_vis = get_vis(g, "Text")
    g_vis.save_graph("text2graph.html")

    g = Graph()
    g.parse(outputFolder + "/image2graph/input_paper/image2graph.ttl", format="ttl")
    g_vis = get_vis(g, "Image")
    g_vis.save_graph("image2graph.html")
    
    g = Graph()
    code_dir_name = os.listdir(codeFolder)[0]
    g.parse(outputFolder + "/code2graph/" + code_dir_name + "/code2graph.ttl", format="ttl")
    # rels = ["https://github.com/deepcurator/DCC/calls", "https://github.com/deepcurator/DCC/followedBy"]
    g_vis = get_vis(g, "Code", rels=rels, show=True)
    g_vis.show("code2graph.html")
    
    # vis_file_to_display = outputFolder + "/code2graph/pointnetvlad-master/pointnetvlad_clsquadruplet_loss.html"
    # iframe = '<iframe src=' + vis_file_to_display + ' width=100% height=350></iframe>'
    # IPython.display.HTML(iframe)


def addTitleToHTML(html_file, title):
    newFileContent = ""
    with open(html_file, "r") as f:
        for line in f:
            if line.strip() == "<body>":
                newFileContent += "<body><h2 style=\"font-family: Arial;\">" + title + "</h2>\n"
            else:
                newFileContent += line
    
    with open(html_file, "w") as f:
        f.write(newFileContent)          

def vis(rels):
    visualize_graphs(rels)
    
    text_file = "text2graph.html"
    addTitleToHTML(text_file, "Text2Graph")
    text_iframe = '<iframe src=' + text_file + ' width=100% height=520></iframe>'
    
    img_file = "image2graph.html"
    addTitleToHTML(img_file, "Image2Graph")
    img_iframe = '<iframe src=' + img_file + ' width=100% height=520></iframe>'
    
    # code_file = outputFolder + "/code2graph/pointnetvlad/pointnetvlad_clsquadruplet_loss.html"
    code_file = "code2graph.html"
    addTitleToHTML(code_file, "Code2Graph")
    code_iframe = '<iframe src=' + code_file + ' width=100% height=520></iframe>'
    
    return text_iframe, img_iframe, code_iframe

def run(pdfURL, codeURL):
    # Delete existing folders and their contents
    if os.path.exists(inputFolder):
        shutil.rmtree(inputFolder,ignore_errors=False,onerror=onerror)
    if os.path.exists(codeFolder):
        shutil.rmtree(codeFolder,ignore_errors=False,onerror=onerror)
    if os.path.exists(outputFolder):
        shutil.rmtree(outputFolder,ignore_errors=False,onerror=onerror)
    if os.path.exists(inputCSV):
        os.remove(inputCSV)

    # Create new folders
    os.makedirs(inputFolder)
    os.makedirs(codeFolder)
    os.makedirs(outputFolder)
    
    # Uncomment below two lines if you have a proxy in your network
    os.environ['http_proxy'] = ""
    os.environ['https_proxy'] = ""
    
    # Download paper pdf
    download_file(pdfURL)
    
    # Download paper code
    git.Git(codeFolder).clone(codeURL)
    # Get code folder name
    code_dir_name = os.listdir(codeFolder)[0]
    # code_dir_name = "input_paper"
    # print("DIR NAME: " + code_dir_name)
    
    enablePrint()
    
    print("Retrieved paper and code.")
    
    disablePrint()
    
    # Create CSV file
    with open(inputCSV, 'w') as f:
        f.write('paper,code\n')
        f.write('input_paper.pdf,' + code_dir_name)
    
    # Uncomment below two lines if you have a proxy in your network
    os.environ['http_proxy'] = ""
    os.environ['https_proxy'] = ""

    t2graph.run(inputFolder, outputFolder, ontology_file, text2graph_models_dir, grobid_client)
    
    i2graph.run(inputFolder, outputFolder, ontology_file, image2graph_models_dir)
    
    c2graph.run(codeFolder, outputFolder, ontology_file, inputCSV)
    
    # enablePrint()
    
    print("Completed")

import copy, ast, astor, pprint
import networkx as nx

from glob import glob
from pathlib import Path
from rdflib import Graph, BNode, RDF, RDFS, URIRef, Literal, OWL, XSD
from rdflib.resource import Resource
from pyvis.network import Network

import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import matplotlib.pyplot as plt

# from code2graph.core.pyan.node import Flavor

# cnames = ['blue', 'green', 'red', 'cyan', 'orange', 'black', 'purple', 'yellow', 'navy', "brown", "gray", "magenta", "#ee55ff", 'blue', 'green', 'red', 'cyan', 'orange', 'black', 'purple', 'yellow', 'navy', "brown", "gray", "magenta", "#ee55ff"]
cnames = ['#a569bd', '#ec7063', '#5dade2', '#58d68d', '#f4d03f', '#bfc9ca', '#d0efff', '#2a9df4', '#45b6fe', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573', '#566573']

auto_color_map_merged = {}
auto_color_map_merged["https://cso.kmi.open.ac.uk/"] = cnames[0]
auto_color_map_merged["https://github.com/deepcurator/DCC/Eval"] = "#a1513b"
auto_color_map_merged["https://github.com/deepcurator/DCC/Generic"] = cnames[2]
auto_color_map_merged["https://github.com/deepcurator/DCC/Publication"] = cnames[1]
auto_color_map_merged["https://github.com/deepcurator/DCC/Method"] = cnames[3]
auto_color_map_merged["https://github.com/deepcurator/DCC/Task"] = cnames[4]
auto_color_map_merged["https://github.com/deepcurator/DCC/Other"] = cnames[5]
auto_color_map_merged["https://github.com/deepcurator/DCC/Material"] = cnames[6]
auto_color_map_merged["https://github.com/deepcurator/DCC/Figure"] = cnames[3]
auto_color_map_merged["https://github.com/deepcurator/DCC/RnnBlock"] = cnames[1]
auto_color_map_merged["https://github.com/deepcurator/DCC/NormBlock"] = "#f01f7a"
auto_color_map_merged["https://github.com/deepcurator/DCC/UserDefined"] = cnames[4]
auto_color_map_merged["https://github.com/deepcurator/DCC/tf"] = cnames[3]
auto_color_map_merged["Literal"] = cnames[5]
auto_color_map_merged["Text"] = "#83de1b"

defaults = []

def addNode(G, src, dst, edge, src_type, dst_type, src_text, dst_text, src_title, dst_title, color_map):
    """if ".txt" in str(src):
        start_index = str(src).rfind(".txt") + 5
        src_label = str(src)[start_index:]
    else:
        start_index = str(src).rfind("/") + 1
        src_label = str(src)[start_index:]
    
    if ".txt" in str(dst):
        start_index = str(dst).rfind(".txt") + 5
        dst_label = str(dst)[start_index:]
    else:
        start_index = str(dst).rfind("/") + 1
        dst_label = str(dst)[start_index:]"""
    
    edge_width = 0.5
    dash = False
    if str(edge) == "https://github.com/deepcurator/DCC/followedBy":
        dash = True     
    
    if color_map == None:
        if src_type not in auto_color_map:
            auto_color_map[src_type] = cnames[len(auto_color_map.items())]
        if dst_type not in auto_color_map:
            auto_color_map[dst_type] = cnames[len(auto_color_map.items())]

        G.add_node(src, size=10, title=src_title, label=src_text, physics=True, color=auto_color_map[src_type])
        G.add_node(dst, size=10, title=dst_title, label=dst_text, physics=True, color=auto_color_map[dst_type])
        G.add_edge(src, dst, width=edge_width, title=str(edge), physics=True, dashes=dash)
    else:
        if src_type not in color_map:
            # cnames[(len(defaults) % 10) + 5]
            color_map[src_type] = "#ff9a0d"
            defaults.append(src_type)
        if dst_type not in color_map:
            color_map[dst_type] = "#ff9a0d"
            defaults.append(dst_type)

        G.add_node(src, size=10, title=src_title, label=src_text, physics=True, color=color_map[src_type])
        G.add_node(dst, size=10, title=dst_title, label=dst_text, physics=True, color=color_map[dst_type])
        G.add_edge(src, dst, width=edge_width, title=str(edge), physics=True, dashes=dash)       

def get_rels(graph):
    auto_color_map = auto_color_map_ont
    rel_count_map = {}
    for src, edge, dst in graph:
        edge_str = str(edge)
        if edge_str not in rel_count_map:
            rel_count_map[edge_str] = 1
        else:
            c = rel_count_map[edge_str]
            rel_count_map[edge_str] = (c+1)
    
    return rel_count_map

def get_vis(graph, modality, max_node_count=10000, rels=[], show=True, color_map=None, directed=True, notebook=True):
    # ont = Graph()
    # ont.parse("run_all_modalities/DeepSciKG.nt", format="ttl")
            
    G = Network(height="500px", width="100%", directed=directed, notebook=notebook)
    
    auto_color_map = auto_color_map_merged
    if  modality == "Merged":
        G.show_buttons(filter_=['physics'])
    
    count = 0
    rel_list = []
    hasText = URIRef("https://github.com/deepcurator/DCC/hasText")
    for src, edge, dst in graph:
        if edge == hasText:
            continue
        # , "https://github.com/deepcurator/DCC/followedBy"
        # if modality == "Code" and str(edge) not in ["https://github.com/deepcurator/DCC/" + rel]:
          #   continue
        if edge not in rel_list:
            rel_list.append(edge)
        if len(rels) > 0:
            if ((show) and (str(edge) not in rels)) or ((not show) and (str(edge) in rels)):
                continue
        
        src_type = [x for x in graph[src:RDF.type]]
        dst_type = [x for x in graph[dst:RDF.type]]
        
        if modality == "Image" and src_type == []:
            if "_Comp" in str(src):
                src_type = "Component"
            elif "_Text" in str(src):
                src_type = "Text"

        if modality == "Image" and dst_type == []:
            if "_Comp" in str(dst):
                dst_type = "https://github.com/deepcurator/DCC/Component"
            elif "_Text" in str(dst):
                dst_type = "https://github.com/deepcurator/DCC/Text"

        src_text = str(src)
        for s,p,o in graph.triples((src, hasText, None)):
            # print(s,p,o)
            src_text = o
        dst_text = str(dst)
        for s,p,o in graph.triples((dst, hasText, None)):
            # print(s,p,o)
            dst_text = o
        
        src_text = src_text.replace("https://github.com/deepcurator/DCC/", "")
        dst_text = dst_text.replace("https://github.com/deepcurator/DCC/", "")
        
        src_text = src_text.replace("http://www.w3.org/2002/07/owl", "")
        dst_text = dst_text.replace("http://www.w3.org/2002/07/owl", "")    
        
        if len(src_type) > 0:
            src_type = str(src_type[0])
        else:
            if str(src).startswith("http://www.w3.org/2002/07/owl"):
                src_type = "OWL"
            elif str(dst_type) == "http://www.w3.org/2002/07/owl#Class":
                dst_type = "OWL"
                continue
            elif str(src).startswith("https://github.com/deepcurator/DCC/"):
                src_type = str(src)
            elif src_type == []:
                src_type = "Literal"
                continue
            else:
                # src_type = "blank node"
                continue
        if len(dst_type) > 0:
            dst_type = str(dst_type[0])
        else:
            if str(dst).startswith("http://www.w3.org/2002/07/owl"):
                dst_type = "OWL"
                # continue
            elif str(dst_type) == "http://www.w3.org/2002/07/owl#Class":
                dst_type = "OWL"
                continue
            elif str(dst).startswith("https://github.com/deepcurator/DCC/"):
                # print("--------------------" + dst)
                # dst_type = "DCC"
                dst_type = str(dst)
            elif dst_type == []:
                dst_type = "Literal"
                # continue
            else:
                # dst_type = "blank node"
                continue
        
        src_title = str(src)
        dst_title = str(dst)
        
        if str(src).startswith("https://cso.kmi.open.ac.uk/"):
            src_type = "https://cso.kmi.open.ac.uk/"
        # (modality == "Code" or modality == "Merged") and 
        elif str(src_type).startswith("https://github.com/deepcurator/DCC/UserDefined"):
            src_type = "https://github.com/deepcurator/DCC/UserDefined"
            src_text = src_text.split(".")[-1]
        # (modality == "Code" or modality == "Merged") and 
        elif str(src_type).startswith("https://github.com/deepcurator/DCC/tf"):         
            # src = str(src_type)
            src_text = src_type.replace("https://github.com/deepcurator/DCC/", "").split("/")[-1]
            # print(src_type)
            src_title = src_type
            src_type = "https://github.com/deepcurator/DCC/tf"
        elif "_Text" in str(src):
            src_type = "Text"
        
            

        if str(dst).startswith("https://cso.kmi.open.ac.uk/"):
            dst_type = "https://cso.kmi.open.ac.uk/"
        # (modality == "Code" or modality == "Merged") and 
        elif str(dst_type).startswith("https://github.com/deepcurator/DCC/UserDefined"):
            dst_type = "https://github.com/deepcurator/DCC/UserDefined"
            dst_text = dst_text.split(".")[-1]
        # (modality == "Code" or modality == "Merged") and 
        elif str(dst_type).startswith("https://github.com/deepcurator/DCC/tf"):        
            # dst = str(dst_type)
            dst_text = dst_type.replace("https://github.com/deepcurator/DCC/", "").split("/")[-1]
            # print(dst_type)
            dst_title = dst_type
            dst_type = "https://github.com/deepcurator/DCC/tf"
        elif "_Text" in str(dst):
            dst_type = "Text"
       
        src_text = src_text.replace(".txt", "")
        dst_text = dst_text.replace(".txt", "")
        src = str(src).replace(".txt", "")
        dst = str(dst).replace(".txt", "")
        src_title = src_title.replace(".txt", "")
        dst_title = dst_title.replace(".txt", "")
        
        if len(rels) > 0:
            if (show) and (str(edge) in rels):
                addNode(G, src, dst, edge, src_type, dst_type, src_text, dst_text, src_title, dst_title, auto_color_map)
            elif (not show) and (str(edge) not in rels):
                addNode(G, src, dst, edge, src_type, dst_type, src_text, dst_text, src_title, dst_title, auto_color_map)
        else:
            addNode(G, src, dst, edge, src_type, dst_type, src_text, dst_text, src_title, dst_title, auto_color_map)
        if count == max_node_count:
            break  
        count += 1
    
    return G
import sys, ast, astor, os, pdb, pprint
import numpy as np
import networkx as nx

from glob import glob
from pathlib import Path
from rdflib import Graph, BNode, RDFS, RDF, URIRef, Literal
from pyvis.network import Network
from sklearn.preprocessing import LabelEncoder

from ontologymanager import OntologyManager
from pyan.analyzer import CallGraphVisitor
from pyan.node import Flavor

def get_name(node):
    return "%s.%s" % (node.namespace, node.name)

class TFVisitor(ast.NodeVisitor):
    
    def __init__(self, node_name, tf_dict):
        
        self.type_manager = OntologyManager()  
        self.starting_node_name = node_name
        self.triples = []
        
        self.types = tf_dict
        self.unmatched_func_calls = []
    
    def visit_Call(self, node):

        call_name = astor.to_source(node.func).strip()

        matched = False

        if '.' in call_name and 'self' not in call_name:
            
            result = self.type_manager.fuzzy_search(call_name)
            
            if result:

                matching = self.type_manager.type_hash[result[0]]
                prev = self.starting_node_name.split('.')[-1]
                curr = matching['name'].split('.')[-1]
                
                if curr not in self.types:
                    name = prev + "." + curr
                    self.types[curr] = 1
                else:
                    name = "%s.%s_%d" % (prev, curr, self.types[curr])
                    self.types[curr] += 1
                

                self.triples.append(
                    ( BNode(self.starting_node_name),
                      BNode("call"), 
                      BNode(name) ))

                self.triples.append(
                    ( BNode(name), 
                      OntologyManager.is_type, 
                      BNode(matching['url']) ))
                
                if len(node.args):
                    for idx, arg in enumerate(node.args):
                        another_visitor = TFVisitor(name, {})
                        another_visitor.visit(arg)

                        if another_visitor.triples:
                            self.triples += another_visitor.triples
                        else:
                            # print(arg, astor.to_source(arg))
                            if isinstance(arg, ast.Str):
                                self.triples.append(
                                    ( BNode(name), 
                                      BNode("has_arg%d"%idx), 
                                      BNode((arg.s))))
                            elif isinstance(arg, ast.Num):
                                self.triples.append(
                                    ( BNode(name), 
                                      BNode("has_arg%d"%idx), 
                                      BNode((arg.n))))
                            

                    
                if len(node.keywords):
                    for keyword in node.keywords:
                        another_visitor = TFVisitor(name,{})
                        another_visitor.visit(keyword)
                        if another_visitor.triples:
                            self.triples += another_visitor.triples
                        else:
                            # print(keyword.arg, keyword.value, astor.to_source(keyword.value))
                            if isinstance(keyword.value, ast.Str):
                                self.triples.append(
                                    ( BNode(name), 
                                      BNode("has_%s"%str(keyword.arg)), 
                                      BNode(keyword.value.s)))
                            elif isinstance(keyword.value, ast.Num):
                                self.triples.append(
                                    ( BNode(name), 
                                      BNode("has_%s"%str(keyword.arg)), 
                                      BNode(keyword.value.n)))
                            elif isinstance(keyword.value, ast.Attribute):
                                self.triples.append(
                                    ( BNode(name), 
                                      BNode("has_%s"%str(keyword.arg)), 
                                      BNode(astor.to_source(keyword.value))))
                            elif isinstance(keyword.value, ast.Tuple):
                                self.triples.append(
                                    ( BNode(name), 
                                      BNode("has_%s"%str(keyword.arg)), 
                                      BNode(astor.to_source(keyword.value))))
                
                matched = True

        if not matched:
            self.unmatched_func_calls.append(call_name)

            if len(node.args):
                for arg in node.args:
                    self.visit(arg)
            
            if len(node.keywords):
                for keyword in node.keywords:
                    self.visit(keyword)

class CustomVisitor:
    
    def __init__(self, call_graph_visitor):
        self.call_graph_visitor = call_graph_visitor
      
        self.triples = [] # used for adding up the results. 

    def visit(self, node, parent):
        if not node:
            return
        
        tf_dict={}

        # iterate through each body instructions.      
        for child in ast.iter_child_nodes(node):
            # print("instruction", type(child), ":", astor.to_source(child).strip())
            if isinstance(child, (ast.Expr, ast.UnaryOp, ast.withitem, ast.Assign, ast.Compare, \
                                  ast.AugAssign, ast.Return, ast.Call, ast.Assert)):
                # The case of unit scannable instruction
                triples, non_tf_functions = self.scan_instruction(parent, child, tf_dict)              

                self.triples += triples
                for func in non_tf_functions:
                    if parent.split('.')[-1] == func.name: # this is to avoid infinite loops.
                        continue

                    self.visit(func.ast_node, get_name(func))
            
            elif isinstance(child, (ast.If, ast.Try, ast.ExceptHandler, ast.For, ast.With)):
                self.visit(child, parent)
            
            elif isinstance(child, (ast.FunctionDef, ast.arguments)):
                pass
    
    def scan_instruction(self, prev_node, node, curr_tf_dict):
        other_fucntions_to_try = []

        visitor = TFVisitor(prev_node, curr_tf_dict)
        visitor.visit(node)
                
        # deal with mismatching function call 
        for func in visitor.unmatched_func_calls:    
            # print(func, "not matching to tf")
            name = func.split('.')[-1]
            if name not in self.call_graph_visitor.nodes:
                continue

            for func_cand in self.call_graph_visitor.nodes[name]:
                if isinstance(func_cand.ast_node, ast.FunctionDef):
                    other_fucntions_to_try.append(func_cand)

        return visitor.triples, other_fucntions_to_try

class TFTokenExplorer:
    
    def __init__(self, code_repo_path):
        
        self.graphs = []                
        self.code_repo_path = code_repo_path
        
        self.call_graph_visitor = CallGraphVisitor(glob("%s/**/*.py" % str(code_repo_path), recursive=True))

        self.call_graph = Graph() # complete_graph
        self.RDF_dict = {} # hashmap from RDF node name to pyan node.

    def explore_code_repository(self):
        
        self.build_defines()
        self.build_calls()
        self.build_tf_types()
        self.separate_call_path()
        

    def build_defines(self):
        
        self.all_defines = set()
        
        defines_edges = self.call_graph_visitor.defines_edges
        
        for src in defines_edges:
            src_name = get_name(src)
            
            self.call_graph.add((BNode(src_name), OntologyManager.is_type, BNode(src.flavor)))

            for dst in defines_edges[src]:
                dst_name = get_name(dst)
                
                self.call_graph.add((BNode(dst_name), OntologyManager.is_type, BNode(dst.flavor)))

                # self.call_graph.add((BNode(src_name), BNode("defines"), BNode(dst_name)))

                self.all_defines.add(src)
                self.all_defines.add(dst)
    
    def build_calls(self):
        
        for define in self.all_defines:
            if define not in self.call_graph_visitor.uses_edges:
                continue
            calls = self.call_graph_visitor.uses_edges[define]
            src_name = get_name(define)
            
            for call in calls:
                if call.flavor == Flavor.FUNCTION or call.flavor == Flavor.METHOD:
                    dst_name = get_name(call)

                    if src_name != dst_name:
                        self.call_graph.add((BNode(src_name), BNode("call"), BNode(dst_name)))

                        self.RDF_dict[BNode(src_name)]=define
                        self.RDF_dict[BNode(dst_name)]=call

    def build_tf_types(self):
        for node in self.RDF_dict:
            print("Processing node: %s" % str(node))
            visitor = CustomVisitor(self.call_graph_visitor)
            visitor.visit(self.RDF_dict[node].ast_node, node)
        
            self.call_graph += visitor.triples
    
    def separate_call_path(self):
    
        G = nx.DiGraph()

        # generate call graph from the RDF graph
        for s,o in self.call_graph[:BNode("call")]:
            G.add_node(str(s))
            G.add_node(str(o))
            G.add_edge(str(s), str(o))

        # find zero in-degree vertex in call graph
        starts = []
        for v in G.in_degree():
            if v[1] == 0:
                starts.append(v[0])

        # run dfs & bfs from each zero in-degree vertice
        for idx, start in enumerate(starts):
            self.graphs.append( (self.bfs_gen_call_path(start, self.call_graph), 
                                 self.dfs_gen_tf_sequence(start, self.call_graph)) )

    def bfs_gen_call_path(self, root, graph):
        g = Graph()
        
        fringe = [BNode(root)]
        visited = []
        
        while fringe:
            node = fringe.pop(0)

            g += graph.triples((node, None, None))

            visited.append(node)
            
            for o in graph[node:BNode("call")]:
                if not o in visited:
                    g.add((node, BNode("call"), o))
                    fringe.append(o)

        return g

    def dfs_gen_tf_sequence(self, root, graph):

        sequence= [] 

        fringe = [BNode(root)]
        visited = []
        
        while fringe:
            node = fringe.pop(0)

            types = graph.triples((node, OntologyManager.is_type, None))

            is_node_tf_type= False
            
            for s,p,o in types: 
                if "Flavor.FUNCTION" not in str(o) and "Flavor.METHOD" not in str(o) and "Flavor.MODULE" not in str(o):
                    sequence.append(str(o))
                    is_node_tf_type = True
            
            if is_node_tf_type: # should append other attributes
                for s,p,o in graph.triples((node, None, None)):
                    if p is not OntologyManager.is_type:
                        sequence.append(str(o))

            visited.append(node)
            
            for o in graph[node:BNode("call")]:
                if not o in visited:
                    fringe.insert(0, o)

        return sequence

    def pyvis_draw(self, graph, name):       
        
        cnames = ['blue', 'green', 'red', 'cyan', 'orange', 'black', 'purple', 'purple', 'purple']

        G= Network(height="800px", width="70%", directed=True)
        data=[]
        for u,e,v in graph:
            data.append(e)

        label_encoder = LabelEncoder()
        integer_encoded = np.array(label_encoder.fit_transform(data), dtype='float')
        i=0
        for src,e,dst in graph:

            src_type = [x for x in self.call_graph[src:OntologyManager.is_type]]
            dst_type = [x for x in self.call_graph[dst:OntologyManager.is_type]]
            
            if len(src_type):
                if str(Flavor.FUNCTION) == str(src_type[0]) or str(Flavor.METHOD) == str(src_type[0]):
                    G.add_node(src, title=src, physics=True, color=cnames[0])
                elif str(Flavor.CLASS) == str(src_type[0]) or str(Flavor.MODULE) == str(src_type[0]):
                    G.add_node(src, title=src, physics=True, color=cnames[1])
                else:
                    G.add_node(src, title=src, physics=True, color=cnames[2])
            else:
                G.add_node(src, title=src, physics=True, color=cnames[3])

            if len(dst_type):
                if 'FUNCTION' in str(dst_type[0]) or 'METHOD' in str(dst_type[0]):
                    G.add_node(dst, title=dst, physics=True, color=cnames[0])
                elif 'CLASS' in str(dst_type[0]) or 'MODULE' in str(dst_type[0]):
                    G.add_node(dst, title=dst, physics=True, color=cnames[1])
                else:
                    G.add_node(dst, title=dst, physics=True, color=cnames[2])
            else:
                G.add_node(dst, title=dst, physics=True, color=cnames[3])
            
            G.add_edge(src,dst, width=0.5, title=data[i], physics=True)
            
            i+=1

        G.hrepulsion(node_distance=120,
                     central_gravity=0.0,
                     spring_length=100,
                     spring_strength=0.01,
                     damping=0.09)
        
        G.show_buttons(filter_=['physics'])
        G.show("%s.html"%name)
    
    def visualize(self):

        for idx, graph in enumerate(self.graphs): # draw each model graph
            rdf_graph, sequence = graph

            # only show the function calls
            draw_graph = Graph()
            draw_graph += rdf_graph.triples((None, BNode("call"), None))
            self.pyvis_draw(draw_graph, str((Path(".")/"test"/"fashion_mnist"/("light_weight_model%d"%(idx))).absolute()))

            pprint.pprint(sequence)

        self.pyvis_draw(self.call_graph, str((Path(".")/"test"/"fashion_mnist"/"light_weight_complete").absolute())) # the complete graph 

if __name__ == "__main__":
    
    # path = Path("..")/"data"/"data_paperswithcode"/"24"/"FewShot_GAN-Unet3D-master"
    # path = Path("..")/"data"/"data_paperswithcode"/"30"/"adaptive-f-divergence-master"
    path = Path(".")/"test"/"fashion_mnist"

    e = TFTokenExplorer(path)

    e.explore_code_repository()
    e.visualize()

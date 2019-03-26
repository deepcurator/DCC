## graphlightweight.py
##
## Utility function to extract architecturally related information from the
## given codes of a deep learning architecture.
## Static code analysis methods have been used, pyan library, AST library
## Pyan Library by: Edmund Horner
## repository link: https://github.com/davidfraser/pyan
## library web-site: https://ejrh.wordpress.com/2012/01/31/call-graphs-in-python-part-2/
##
## This program has been created by University of California, Irvine
## AICPS(Advanced Integrated Cyber-Physical Systems) Lab Members:
## Sujit Rokka Chhetri
## Shih-Yuan Yu
## Ahmet Salih Aksakal
##
## Advised by Professor Mohammad Al Faruque
##
##
### This material is based upon work supported by the
###	Defense Advanced Research Projects Agency (DARPA)
### under Agreement No. HR00111990010

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

class CallVisitor(ast.NodeVisitor):

    def __init__(self, call_graph_visitor, parent):

        self.call_graph_visitor = call_graph_visitor
        self.type_manager = OntologyManager()
        self.root = parent

        self.function_to_be_visited=[]

    def visit_Call(self, node):

        call_name = astor.to_source(node.func).strip()
        base_name = call_name.split('.')[-1]

        # print("finding call full name: %s, base name: %s" % (call_name, base_name))
        # print(self.root)
        proj_function_found = False
        matched = False

        if "pyan" in self.root:
            for cand in self.call_graph_visitor.uses_edges[self.root["pyan"]]:
                if isinstance(cand.ast_node, ast.FunctionDef):
                    if cand.name == base_name:
                        # print (cand)

                        new_node = {"name":base_name, "children":[], "type": cand.flavor, "pyan":cand}
                        self.root["children"].append(new_node)

                        self.function_to_be_visited.append((cand, new_node))

                        proj_function_found = True

                    elif cand.name == "__init__" and base_name == cand.namespace.split('.')[-1]:

                        new_node = {"name":base_name+".__init__", "children":[], "type": cand.flavor, "pyan":cand}
                        self.root["children"].append(new_node)

                        self.function_to_be_visited.append((cand, new_node))

                        proj_function_found = True

        if not proj_function_found and '.' in call_name:

            result = self.type_manager.fuzzy_search(call_name)

            if result:
                matching = self.type_manager.type_hash[result[0]]
                # print(call_name, matching)
                new_node = {"name":matching['name'].split('.')[-1], "url":matching['url'], "children":[], "type":"tf_keyword"}

                self.root['children'].append(new_node)

                if len(node.args):
                    for idx, arg in enumerate(node.args):
                        another_visitor = CallVisitor(self.call_graph_visitor, new_node)
                        another_visitor.visit(arg)

                        new_node["args"] = []
                        if isinstance(arg, ast.Str):
                            new_node["args"].append(arg.s)
                        elif isinstance(arg, ast.Num):
                            new_node["args"].append(arg.n)

                if len(node.keywords):
                    for keyword in node.keywords:
                        another_visitor = CallVisitor(self.call_graph_visitor, new_node)
                        another_visitor.visit(keyword)

                        new_node["keywords"] = {}
                        if isinstance(keyword.value, ast.Str):
                            new_node["keywords"][str(keyword.arg)] = keyword.value.s
                        elif isinstance(keyword.value, ast.Num):
                            new_node["keywords"][str(keyword.arg)] = keyword.value.n
                        elif isinstance(keyword.value, ast.Attribute):
                            new_node["keywords"][str(keyword.arg)] = astor.to_source(keyword.value).strip()
                        elif isinstance(keyword.value, ast.Tuple):
                            new_node["keywords"][str(keyword.arg)] = astor.to_source(keyword.value).strip()
                matched = True

        if not matched:
            if len(node.args):
                for arg in node.args:
                    another_visitor = CallVisitor(self.call_graph_visitor, self.root)
                    another_visitor.visit(arg)

                    self.function_to_be_visited += another_visitor.function_to_be_visited

            if len(node.keywords):
                for keyword in node.keywords:
                    another_visitor = CallVisitor(self.call_graph_visitor, self.root)
                    another_visitor.visit(keyword)

                    self.function_to_be_visited += another_visitor.function_to_be_visited


class ProgramLineVisitor:

    def __init__(self, call_graph_visitor, node):
        self.call_graph_visitor = call_graph_visitor

        self.root = {"name": get_name(node), "children":[], "type": node.flavor, "pyan":node}

    def visit(self, node, parent):

        if node is None:
            return

        for child in ast.iter_child_nodes(node):
            # print("instruction", type(child), ":", astor.to_source(child).strip())

            if isinstance(child, (ast.Expr, ast.UnaryOp, ast.withitem, ast.Assign, ast.Compare, \
                                  ast.AugAssign, ast.Return, ast.Call, ast.Assert)):
                # The case of unit scannable instruction
                # print("instruction", type(child), ":", astor.to_source(child).strip())

                call_visitor = CallVisitor(self.call_graph_visitor, parent)
                call_visitor.visit(child) # search function calls by line

                for function, function_node in call_visitor.function_to_be_visited:
                    # print("revisit function: ", function_node)
                    # if "pyan" in parent:
                    #     print(parent["pyan"])
                    #     print(self.call_graph_visitor.uses_edges[parent["pyan"]])
                    if function_node['name'] != parent['name']:
                        self.visit(function.ast_node, function_node)

            elif isinstance(child, (ast.If, ast.Try, ast.ExceptHandler, ast.For, ast.With)):
                self.visit(child, parent)

            elif isinstance(child, (ast.FunctionDef, ast.arguments)):
                pass

class TFTokenExplorer:

    def __init__(self, code_repo_path):

        self.code_repo_path = code_repo_path

        self.call_graph_visitor = CallGraphVisitor(glob("%s/**/*.py" % str(code_repo_path), recursive=True))
        # pprint.pprint(self.call_graph_visitor.nodes)
        # generated in build the one complete call graph
        self.call_graph = Graph() # complete_graph
        self.pyan_node_dict = {} # hashmap from RDF node name to pyan node.

        # generated in build call trees
        self.call_trees = {}
        self.call_tree_graphs = []

        # generated in build rdf graphs
        self.rdf_graphs = {}

        # generated in build tfseqeuences
        self.tfseqeuences = {}

    def explore_code_repository(self):

        self.build_call_graph()
        self.build_call_trees()
        self.build_rdf_graphs()
        self.build_tfsequences()

        self.dump_information()


    def build_call_graph(self):
        for caller in self.call_graph_visitor.uses_edges:
            if caller.flavor is not Flavor.UNSPECIFIED and caller.flavor is not Flavor.UNKNOWN:

                self.call_graph.add((BNode(get_name(caller)), OntologyManager.is_type, BNode(caller.flavor)))
                self.pyan_node_dict[get_name(caller)]=caller

                for callee in self.call_graph_visitor.uses_edges[caller]:
                    if callee.flavor is not Flavor.UNSPECIFIED and callee.flavor is not Flavor.UNKNOWN:
                        self.call_graph.add((BNode(get_name(callee)), OntologyManager.is_type, BNode(callee.flavor)))
                        self.pyan_node_dict[get_name(callee)]=callee

                        self.call_graph.add((BNode(get_name(caller)), OntologyManager.call, BNode(get_name(callee))))

    def build_call_trees(self):
        roots = self.find_roots(self.call_graph)

        for root in roots:
            call_tree = self.grow_function_calls(root)

            call_tree["rdf_name"]= call_tree["name"]

            self.populate_call_tree(call_tree)

            self.call_trees[root] = call_tree


    def find_roots(self, graph):

        G = nx.DiGraph()

        for s,o in graph[:OntologyManager.call]:
            G.add_node(str(s))
            G.add_node(str(o))
            G.add_edge(str(s), str(o))

        starts = []
        for v in G.in_degree():
            if v[1] == 0:
                starts.append(v[0])

        return starts

    def grow_function_calls(self, start):

        print("Processing node: %s" % str(start))
        line_visitor = ProgramLineVisitor(self.call_graph_visitor, self.pyan_node_dict[start])
        line_visitor.visit(self.pyan_node_dict[start].ast_node, line_visitor.root)

        return line_visitor.root

    def populate_call_tree(self, node):

        if "children" in node:
            for idx, child in enumerate(node["children"]):
                child["rdf_name"] = node["rdf_name"] + "." + child["name"] + "_" + str(idx)
                self.populate_call_tree(child)

    def build_rdf_graphs(self):
        for root in self.call_trees:
            graph = Graph()
            self.build_rdf_graph(self.call_trees[root], graph)
            self.rdf_graphs[root] = graph

    def build_rdf_graph(self, node, graph): # need to use DFS (might encounter max recursion limit problem)

        if "type" in node:
            if node["type"] == "tf_keyword":
                graph.add((BNode(node['rdf_name']), OntologyManager.is_type, BNode(node["url"])))
            else:
                graph.add((BNode(node['rdf_name']), OntologyManager.is_type, BNode(node["type"])))

        if "children" in node:
            for idx, child in enumerate(node["children"]):
                graph.add((BNode(node['rdf_name']), OntologyManager.call, BNode(child["rdf_name"])))
                # if idx > 0:
                    # graph.add((BNode(node["children"][idx-1]['rdf_name']), BNode("followed_by"), BNode(child["rdf_name"])))
                self.build_rdf_graph(child, graph)

        if "args" in node:
            for idx, arg in enumerate(node['args']):
                graph.add((BNode(node['rdf_name']), BNode("has_arg%d"%idx), BNode((arg))))

        if "keywords" in node:
            for keyword in node['keywords']:
                graph.add((BNode(node['rdf_name']), BNode("has_%s"%str(keyword)), BNode(node['keywords'][keyword])))

    def build_tfsequences(self):
        for root in self.call_trees:
            sequence = []
            self.build_tfsequence(self.call_trees[root], sequence)
            self.tfseqeuences[root] = sequence

    def build_tfsequence(self, node, sequence):

        if "type" in node:
            if node["type"] == "tf_keyword":
                sequence.append(node["name"])
        if "args" in node:
            for idx, arg in enumerate(node['args']):
                sequence.append(arg)
        if "keywords" in node:
            for keyword in node['keywords']:
                sequence.append(node['keywords'][keyword])
        if "children" in node:
            for child in node["children"]:
                self.build_tfsequence(child, sequence)

    def dump_information(self):

        # self.dump_call_graph()
        # self.dump_call_trees()
        # self.dump_rdf_graphs()
        self.dump_tfseuqences()

    def dump_call_graph(self):
        self.pyvis_draw(self.call_graph, str(self.code_repo_path/"call_graph"))
        pprint.pprint(self.pyan_node_dict)

    def dump_call_trees(self):
        print("Dump Code Trees")
        for root in self.call_trees:
            print("for call tree starting from %s"%root)
            pprint.pprint(self.call_trees[root])

    def dump_rdf_graphs(self):
        for root in self.rdf_graphs:
            self.pyvis_draw(self.rdf_graphs[root], str(self.code_repo_path/root.replace('.','')))

    def dump_tfseuqences(self):
        pprint.pprint(self.tfseqeuences)

    def pyvis_draw(self, graph, name):

        cnames = ['blue', 'green', 'red', 'cyan', 'orange', 'black', 'purple', 'purple', 'purple']

        G= Network(height="800px", width="70%", directed=True)

        for src, edge, dst in graph:

            if edge == OntologyManager.is_type:
                continue

            src_type = [x for x in graph[src:OntologyManager.is_type]]
            dst_type = [x for x in graph[dst:OntologyManager.is_type]]

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
                if str(Flavor.FUNCTION) == str(dst_type[0]) or str(Flavor.METHOD) == str(dst_type[0]):
                    G.add_node(dst, title=dst, physics=True, color=cnames[0])
                elif str(Flavor.CLASS) == str(dst_type[0]) or str(Flavor.MODULE) == str(dst_type[0]):
                    G.add_node(dst, title=dst, physics=True, color=cnames[1])
                else:
                    G.add_node(dst, title=dst, physics=True, color=cnames[2])
            else:
                G.add_node(dst, title=dst, physics=True, color=cnames[3])

            G.add_edge(src,dst, width=0.5, title=str(edge), physics=True)

        G.hrepulsion(node_distance=120,
                     central_gravity=0.0,
                     spring_length=100,
                     spring_strength=0.01,
                     damping=0.09)

        G.show_buttons(filter_=['physics'])
        G.show("%s.html"%name)

def lightweight(path):

    e = TFTokenExplorer(path)

    e.explore_code_repository()

if __name__ == "__main__":


    # path = Path(".")/"test"/"FewShot_GAN-Unet3D-master"
    # path = Path(".")/"test"/"adaptive-f-divergence-master"/'bnn'
    # path = Path("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/09/NPRF-master") # python 2
    # path = Path("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/23/KGD-master") # lambda problem
    # path = Path("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/24/FewShot_GAN-Unet3D-master")
    # path = Path("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/31/RecurJac-Jacobian-Bounds-master") # remove parse_landscape.py
    # path = Path("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/34/enas-master") # python 2 
    # path = Path("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/37/Micro-Net-master") # ok
    # path = Path("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/42/AdverseBiNet-master") # ok
    # path = Path("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/43/T-GANs-master") # python 2
    # path = Path("/Users/louisccc/Dropbox/louis_research/development/darpa/DARPA-DCC/data/data_paperswithcode/49/ambient-gan-master") # python 2
    path = Path(".")/"test"/"fashion_mnist"
    path = path.resolve()
    lightweight(path)

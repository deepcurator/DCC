# graphlightweight.py
##
# Utility function to extract architecturally related information from the
# given codes of a deep learning architecture.
# Static code analysis methods have been used, pyan library, AST library
# Pyan Library by: Edmund Horner
# repository link: https://github.com/davidfraser/pyan
# library web-site: https://ejrh.wordpress.com/2012/01/31/call-graphs-in-python-part-2/
##
# This program has been created by University of California, Irvine
# AICPS(Advanced Integrated Cyber-Physical Systems) Lab Members:
# Sujit Rokka Chhetri
# Shih-Yuan Yu
# Ahmet Salih Aksakal
##
# Advised by Professor Mohammad Al Faruque
##
##
# This material is based upon work supported by the
# Defense Advanced Research Projects Agency (DARPA)
# under Agreement No. HR00111990010

import ast
import astor
import pprint
import networkx as nx

from glob import glob
from pathlib import Path
from rdflib import Graph, BNode, RDFS, RDF, URIRef, Literal
from pyvis.network import Network

from .ontologymanager import OntologyManager
from .pyan.analyzer import CallGraphVisitor
from .pyan.node import Flavor


def get_name(node):
    return "%s.%s" % (node.namespace, node.name)

type_manager = OntologyManager()

class CallVisitor(ast.NodeVisitor):

    

    def __init__(self, pyan_edges, parent):

        self.pyan_edges = pyan_edges

        self.root = parent

        self.function_to_be_visited = []

    def check_internal_function(self, base_name):
        if "pyan" in self.root:
            # print("Inside Pyan!!")
            for cand in self.pyan_edges[self.root["pyan"]]:
                # print("Inside Pyan:", cand)
                if isinstance(cand.ast_node, ast.FunctionDef):
                    if cand.name == base_name:
                        # print (cand)

                        new_node = {"name": base_name, "children": [],
                                    "type": "internal_"+cand.flavor.value, "pyan": cand}

                        self.root["children"].append(new_node)

                        self.function_to_be_visited.append((cand, new_node))

                        return True

                    elif cand.name == "__init__" and base_name == cand.namespace.split('.')[-1]:

                        new_node = {"name": base_name+".__init__",
                                    "children": [], "type": "internal_"+cand.flavor.value, "pyan": cand}
                        self.root["children"].append(new_node)

                        self.function_to_be_visited.append((cand, new_node))

                        return True
                elif isinstance(cand.ast_node, ast.FunctionDef):
                    pass
        return False

    def check_tensorflow_function(self, call_name, node):
        # print("inside Call name .!!")
        result = type_manager.fuzzy_search(call_name)
        # print("\ncall_name:", call_name)
        # print("match list:", result)

        if result:
            matching = type_manager.type_hash[result[0]]
            # print(call_name, matching)
            new_node = {"name": matching['name'].split(
                '.')[-1], "url": matching['uri'], "children": [], "type":
                "tf_keyword", "args": [], "keywords": {}}
            new_node["args"] = []

            self.check_args(node, new_node)

            self.check_keywords(node, new_node)

            self.root['children'].append(new_node)
            return True
        else:
            if isinstance(node.func, ast.Call):
                func_visitor = CallVisitor(
                    self.pyan_edges, self.root)
                func_visitor.visit(node.func)
                # import pdb; pdb.set_trace()
                if call_name.split('(')[0] == self.root['children'][-1]['name']:

                    new_node = {
                        "name": call_name, "url": self.root['children'][-1]['uri'], "children": [],
                        "type": self.root['children'][-1]['type'], "args": [], "keywords": {}}

                    self.check_args(node, new_node)

                    self.check_keywords(node, new_node)

                    self.root['children'][-1]['children'].append(new_node)
                    return True
        return False

    def visit_Call(self, node):
        # print("Entered visit Call!")
        call_name = astor.to_source(node.func).strip()
        base_name = call_name.split('.')[-1]
        # print("\n Node:", node)
        # print("finding call full name: %s, base name: %s" %(call_name, base_name))
        # pprint.pprint(astor.dump_tree(node))
        # print(self.root)
        # import pdb; pdb.set_trace()
        if base_name == "__init__":
            base_name = call_name.split('.')[-2] + "." + base_name

        proj_function_found = self.check_internal_function(base_name)

        # TODO: removed '.' in call_name condition, functions like print are on graph
        # TODO: find better ways to blacklist functions
        matched = False
        if not proj_function_found and 'print' not in call_name:
            matched = self.check_tensorflow_function(call_name, node)

        if not matched:
            # print("Inside Not matched!")
            if len(node.args):
                for _, arg in enumerate(node.args):
                    # print("\targs:", arg)
                    another_visitor = CallVisitor(
                        self.pyan_edges, self.root)
                    another_visitor.visit(arg)
                    self.function_to_be_visited += another_visitor.function_to_be_visited

            if len(node.keywords):
                for _, keyword in enumerate(node.keywords):
                    # print("key #:%d keyword name: %s  "%(i, keyword.arg))
                    # print("keyword class:",keyword.__class__.__name__)
                    for _, value in ast.iter_fields(keyword):
                        # print("fieldd:",field, "value:",value)
                        if isinstance(value, ast.Call):
                            another_visitor_k = CallVisitor(
                                self.pyan_edges, self.root)
                            another_visitor_k.visit(keyword)
                            self.function_to_be_visited += another_visitor_k.function_to_be_visited

                        elif isinstance(value, ast.Str):
                            result = type_manager.fuzzy_search(value.s)
                            if result:
                                matching = type_manager.type_hash[result[0]]
                                new_node = {"name": matching['name'].split('.')[-1], "url": matching['uri'],
                                            "children": [], "args": [], "type": "tf_keyword"}
                                self.root['children'].append(new_node)

                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, ast.Str):
                                    result = type_manager.fuzzy_search(
                                        item.s)
                                    if result:
                                        matching = type_manager.type_hash[result[0]]
                                        new_node = {"name": matching['name'].split('.')[-1], "url": matching['uri'],
                                                    "children": [], "args": [], "type": "tf_keyword"}
                                        self.root['children'].append(new_node)

    def check_args(self, node, new_node):
        if len(node.args):
            for _, arg in enumerate(node.args):

                another_visitor = CallVisitor(
                    self.pyan_edges, new_node)
                another_visitor.visit(arg)

                if isinstance(arg, ast.Str):
                    new_node["args"].append(arg.s)
                elif isinstance(arg, ast.Num):
                    new_node["args"].append(arg.n)
                    # print("idx:", idx, "args:", arg.n)
                # TODO: reconsider adding variable name
                # elif isinstance(arg, ast.Name):
                #     new_node["args"].append(arg.id)
                elif isinstance(arg, ast.Tuple):
                    new_node["args"].append(astor.to_source(arg).strip())
                    # print("idx:", idx, "args:", astor.to_source(arg).strip())

    def check_keywords(self, node, new_node):
        if len(node.keywords):
            # import pdb; pdb.set_trace()
            for keyword in node.keywords:
                another_visitor = CallVisitor(
                    self.pyan_edges, new_node)
                another_visitor.visit(keyword)

                if isinstance(keyword.value, ast.Str):
                    new_node["keywords"][str(keyword.arg)] = keyword.value.s
                    # print("--match with .: Keyword", keyword.value.s)
                elif isinstance(keyword.value, ast.Num):
                    new_node["keywords"][str(keyword.arg)] = keyword.value.n
                    # print("--match with .: Keyword", keyword.value.n)
                elif isinstance(keyword.value, ast.NameConstant):
                    new_node["keywords"][str(keyword.arg)] = str(
                        keyword.value.value)
                elif isinstance(keyword.value, ast.Attribute):
                    new_node["keywords"][str(keyword.arg)] = astor.to_source(
                        keyword.value).strip()
                    # print("--match with .: Keyword", astor.to_source(keyword.value).strip())
                elif isinstance(keyword.value, ast.Tuple):
                    new_node["keywords"][str(keyword.arg)] = astor.to_source(
                        keyword.value).strip()
                    # print("--match with .: Keyword", astor.to_source(keyword.value).strip())


class ProgramLineVisitor:

    def __init__(self, pyan_edges, node):
        self.pyan_edges = pyan_edges

        self.root = {"name": get_name(
            node), "children": [], "type": node.flavor, "pyan": node}

    def visit(self, node, parent):

        if node is None:
            return

        for child in ast.iter_child_nodes(node):
            # print("parent:", parent, "child:",child)
            if isinstance(child, (ast.Expr, ast.UnaryOp, ast.withitem, ast.Assign, ast.Compare,
                                  ast.AugAssign, ast.Return, ast.Call, ast.Assert)):

                call_visitor = CallVisitor(self.pyan_edges, parent)
                call_visitor.visit(child)  # search function calls by line

                for function, function_node in call_visitor.function_to_be_visited:
                    if function_node['name'] != parent['name']:
                        self.visit(function.ast_node, function_node)

            elif isinstance(child, (ast.If, ast.Try, ast.ExceptHandler, ast.For, ast.With)):
                self.visit(child, parent)

            elif isinstance(child, ast.FunctionDef):
                pass
            elif isinstance(child, ast.arguments):
                # node["args"].append(child.n)
                pass


class TFTokenExplorer:

    def __init__(self, code_path, config):
        """Initializing the class"""

        self.dump_functions = {
            1: self.dump_call_graph,
            2: self.dump_call_trees,
            3: self.dump_rdf_graphs,
            4: self.dump_tfsequences,
            5: self.dump_rdf_triples,
            6: self.dump_rdf
        }

        self.config = config

        self.code_repo_path = Path(code_path).resolve()
        self.all_py_files = glob("%s/**/*.py" %
                                 str(self.code_repo_path), recursive=True)

        self.call_graph = Graph()  # generated in build the one complete call graph
        self.call_trees = {}      # generated in build call trees
        self.rdf_graphs = {}      # generated in build rdf graphs
        self.rdf_quads = {}      # generated in build rdf graphs
        self.tfsequences = {}      # generated in build tfseqeuences

    def build_pyan_call_graph(self):
        self.call_graph_visitor = CallGraphVisitor(self.all_py_files)

        self.pyan_edges = self.call_graph_visitor.uses_edges
        ''' pyan edges are all relations extracted from pyan, each of element is stored as Node instance. 
            if the node is recognized as an internal function, its flavor will then be 'function'. 
            if the node is recognized as a module, its flavor will then be 'module'.
            
            Example: 
            {<Node module:testGraph_extensive>: {<Node ---:*.fit>,
                                     <Node ---:*.np>,
                                     <Node ---:*.compile>,
                                     <Node ---:*.plt>,
                                     <Node ---:*.Sequential>,
                                     <Node ---:*.TensorBoard>,
                                     <Node function:testGraph_extensive.clearLogFolder>,
                                     <Node ???:*.fashion_mnist>,
                                     <Node ???:*.load_data>,
                                     <Node ???:*.AdamOptimizer>,
                                     <Node ???:*.print>,
                                     <Node ???:*.Flatten>,
                                     <Node ???:*.Dense>,
                                     <Node ---:*.os>,
                                     <Node ---:*.time>,
                                     <Node ---:*.tensorflow>,
                                     <Node ---:*.tf>,
                                     <Node ---:*.evaluate>,
                                     <Node ---:*.tensorflow.keras.callbacks>,
                                     <Node ???:*.relu>,
                                     <Node ???:*.softmax>},
        '''
        self.pyan_node_dict = {}

        ''' 
            pyan_node_dict stores the hashmap from RDF node name to pyan node.
            name -> pyan node.
            Ex. .testGraph_extensive -> <Node module:testGraph_extensive>
        '''

    def explore_code_repository(self):
        self.build_pyan_call_graph()
        self.build_call_graph()
        self.build_call_trees()
        self.recognize_tf_calls()
        self.build_rdf_graphs()
        self.build_tfsequences()

        self.dump_information()

    def recognize_tf_calls(self):
        pass

    def build_call_graph(self):

        for caller, callees in self.pyan_edges.items():

            if caller.flavor is Flavor.UNSPECIFIED:
                continue
            if caller.flavor is Flavor.UNKNOWN:
                continue

            caller_name = get_name(caller)
            caller_type = caller.flavor

            self.call_graph.add(
                (BNode(caller_name), type_manager.is_type, BNode(caller_type)))

            self.pyan_node_dict[caller_name] = caller

            for callee in callees:

                if callee.flavor is Flavor.UNSPECIFIED:
                    continue
                if callee.flavor is Flavor.UNKNOWN:
                    continue

                callee_name = get_name(callee)
                callee_type = callee.flavor

                self.call_graph.add(
                    (BNode(callee_name), type_manager.is_type, BNode(callee_type)))
                self.call_graph.add(
                    (BNode(caller_name), type_manager.call,    BNode(callee_name)))

                self.pyan_node_dict[callee_name] = callee

    def build_call_trees(self):
        roots = self.find_roots(self.call_graph)

        for root in roots:
            print("Start from root:", self.pyan_node_dict[root])

            call_tree = self.grow_function_calls(root)

            call_tree["rdf_name"] = call_tree["name"]

            self.populate_call_tree(call_tree)

            self.call_trees[root] = call_tree

    def find_roots(self, graph):

        G = nx.DiGraph()

        for s, o in graph[:type_manager.call]:
            G.add_node(str(s))
            G.add_node(str(o))
            G.add_edge(str(s), str(o))

        starts = []
        for v in G.in_degree():
            if v[1] == 0:
                starts.append(v[0])

        return starts

    def grow_function_calls(self, start):
        # TODO self.call_graph_visitor should be removed as only node dict is needed.
        line_visitor = ProgramLineVisitor(
            self.pyan_edges, self.pyan_node_dict[start])
        line_visitor.visit(
            self.pyan_node_dict[start].ast_node, line_visitor.root)

        return line_visitor.root

    def populate_call_tree(self, node):

        if "idx" not in node:
            node["idx"] = "1"
        node["name"] = node["name"].replace(
            '\n', '').replace('\t', '').replace('    ', '')

        if "children" in node:
            for idx, child in enumerate(node["children"]):
                child["rdf_name"] = node["rdf_name"] + \
                    "." + child["name"] + "_" + str(idx)
                child["idx"] = node["idx"] + "-" + str(idx)
                self.populate_call_tree(child)

    def build_rdf_graphs(self):
        for root in self.call_trees:
            graph = Graph()
            quad = []
            self.build_rdf_graph(self.call_trees[root], graph, quad, root)
            self.rdf_graphs[root] = graph
            self.rdf_quads[root] = quad

    # need to use DFS (might encounter max recursion limit problem)
    def build_rdf_graph(self, node, graph, quad, root):

        if node["name"] != root:
            quad.append(node["name"] + "\t" + "has_root" +
                        "\t" + root + "\t" + node["idx"] + "\n")

        if "name" in node:
            graph.add(
                (BNode(node["rdf_name"]), BNode("has_function"), BNode(node["name"])))

        if "type" in node:
            if node["type"] == "tf_keyword":
                graph.add(
                    (BNode(node["rdf_name"]), type_manager.is_type, BNode(node["url"])))
                quad.append(node["name"] + "\t" + "is_type" +
                            "\t" + node["url"] + "\t" + node["idx"] + "\n")
            else:
                if isinstance(node["type"], Flavor):
                    node["type"] = node["type"].value
                graph.add(
                    (BNode(node["rdf_name"]), type_manager.is_type, BNode(node["type"])))
                quad.append(node["name"] + "\t" + "is_type" +
                            "\t" + node["type"] + "\t" + node["idx"] + "\n")

        if "children" in node:
            for idx, child in enumerate(node["children"]):
                graph.add(
                    (BNode(node["rdf_name"]), type_manager.call, BNode(child["rdf_name"])))
                quad.append(node["name"] + "\t" + "call" + "\t" +
                            child["name"] + "\t" + node["idx"] + "\n")
                if idx > 0:
                    graph.add((BNode(node["children"][idx-1]["rdf_name"]),
                               BNode("followed_by"), BNode(child["rdf_name"])))
                    quad.append(node["children"][idx-1]["name"] + "\t" + "followed_by"
                                + "\t" + child["name"] + "\t" + node["idx"] + "\n")
                self.build_rdf_graph(child, graph, quad, root)

        if "args" in node and self.config.show_arg:
            # print("\n Node:---->",node, node['args'])
            if len(node['args']) == 3:
                k_size = "("+str(node['args'][1])+","+str(node['args'][2])+")"
                graph.add((BNode(node["rdf_name"]), BNode(
                    "has_output_feature_size"), BNode((node['args'][0]))))
                graph.add((BNode(node["rdf_name"]), BNode(
                    "has_kernel_size"), BNode(k_size)))
                quad.append(node["name"] + "\t" + "has_output_feature_size" +
                            "\t" + node["args"][0] + "\t" + node["idx"] + "\n")
                quad.append(node["name"] + "\t" + "has_kernel_size" +
                            "\t" + k_size + "\t" + node["idx"] + "\n")
            else:
                for idx, arg in enumerate(node['args']):
                    # print(arg)
                    graph.add((BNode(node["rdf_name"]), BNode(
                        "has_arg%d" % idx), BNode((arg))))
                    quad.append(node["name"] + "\t" + ("has_arg%d" % idx) + "\t" + str(arg).replace(
                        '\n', '').replace('\t', '').replace('    ', '') + "\t" + node["idx"] + "\n")

        if "keywords" in node and self.config.show_arg:
            for keyword in node['keywords']:
                graph.add((BNode(node["rdf_name"]), BNode(
                    "has_%s" % str(keyword)), BNode(node['keywords'][keyword])))
                quad.append(node["name"] + "\t" + ("has_%s" % str(keyword)) +
                            "\t" + str(node['keywords'][keyword]) + "\t" + node["idx"] + "\n")

    def build_tfsequences(self):
        for root in self.call_trees:
            sequence = []
            self.build_tfsequence(self.call_trees[root], sequence)
            self.tfsequences[root] = sequence

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
        for option in self.config.output_types:
            print('dump info with option %d: %s' %
                  (option, self.dump_functions[option].__name__))
            self.dump_functions[option]()
        self.dump_rdf_quads()

    def dump_call_graph(self):
        self.pyvis_draw(self.call_graph, str(self.code_repo_path/"call_graph"))
        pprint.pprint(self.pyan_node_dict)

    def dump_call_trees(self):
        print("Dump Code Trees")
        for root in self.call_trees:
            print("for call tree starting from %s" % root)
            pprint.pprint(self.call_trees[root])

    def dump_rdf_graphs(self):
        for root in self.rdf_graphs:
            self.pyvis_draw(self.rdf_graphs[root], str(
                self.code_repo_path/root.replace('.', '')))

    def dump_tfsequences(self):

        pprint.pprint(self.tfsequences)

    def dump_rdf_triples(self):
        combined_triplets_path = str(
            self.code_repo_path/'combined_triples.triples')

        with open(combined_triplets_path, 'w') as combined_file:
            for root in self.rdf_graphs:
                stored_path = str(self.code_repo_path /
                                  (root.replace('.', '') + '.triples'))

                with open(stored_path, 'w') as triplets_file:
                    for sub, pred, obj in self.rdf_graphs[root].triples((None, None, None)):
                        sub = sub.replace('\n', '').replace(
                            '\t', '').replace('    ', '')
                        obj = obj.replace('\n', '').replace(
                            '\t', '').replace('    ', '')
                        triplets_file.write(sub+'\t'+pred+'\t'+obj+'\n')
                        combined_file.write(sub+'\t'+pred+'\t'+obj+'\n')

    def dump_rdf_quads(self):
        combined_quads_path = str(self.code_repo_path/'combined_quads.quads')

        with open(combined_quads_path, 'w') as combined_file:
            for root in self.rdf_quads:
                stored_path = str(self.code_repo_path /
                                  (root.replace('.', '') + '.quads'))
                with open(stored_path, 'w') as quads_file:
                    for quad in self.rdf_quads[root]:
                        quads_file.write(quad)
                        combined_file.write(quad)

    def dump_rdf(self):
        combined_graph = Graph()
        for graph in self.rdf_graphs.values():
            combined_graph += graph
        combined_graph.serialize(destination=str(
            self.code_repo_path / "rdf_graph.rdf"), format='turtle')

    def pyvis_draw(self, graph, name):

        cnames = ['blue', 'green', 'red', 'cyan', 'orange',
                  'black', 'purple', 'purple', 'purple']

        G = Network(height="800px", width="70%", directed=True)

        for src, edge, dst in graph:
            # print(src, edge, dst)

            if edge == type_manager.is_type and not self.config.show_url:
                continue

            src_type = [x for x in graph[src:type_manager.is_type]]
            dst_type = [x for x in graph[dst:type_manager.is_type]]

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

            G.add_edge(src, dst, width=0.5, title=str(edge), physics=True)

        G.hrepulsion(node_distance=120,
                     central_gravity=0.0,
                     spring_length=100,
                     spring_strength=0.01,
                     damping=0.09)

        G.show_buttons(filter_=['physics'])

        # G.show("%s.html" % name)
        G.save_graph("%s.html" % name)

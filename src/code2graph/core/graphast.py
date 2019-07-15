import sys
import ast
import astor
import os
import pdb
import pprint
import numpy as np
import networkx as nx

from glob import glob
from pathlib import Path
from rdflib import Graph, BNode, RDFS, RDF, URIRef, Literal
from pyvis.network import Network
from sklearn.preprocessing import LabelEncoder
from showast.rendering.graphviz import render
from showast import Settings

import itertools

class ASTExplorer:
    
    def __init__(self, code_path):
        # path to code repository
        self.code_path = code_path
        # path of all python files
        self.all_py_files = glob("%s/**/*.py" % str(code_path), recursive=True)
        # leaves nodes of ast tree for each module
        self.leaves={}
        # paths between leaves of ast tree for each module
        self.paths={}
        self.ast_nodes={}
        self.function_defs={}
        self.path_length_upper_bound = 10
        self.path_length_lower_bound = 3

    def process_all_files(self):

        for py_file in self.all_py_files:
            with open(py_file, "rt", encoding="utf-8") as f:
                content = f.read()

                module_node = ast.parse(content, py_file)

                self.leaves[py_file] = self.generate_leaves(module_node)

                self.paths[py_file]  = self.generate_paths(self.leaves[py_file])

                self.ast_nodes[py_file] = module_node
    
                self.function_defs[py_file] = self.generate_function_defs(module_node)
                
                print(py_file, len(self.leaves[py_file]))
        # self.dump_leaf_nodes()
        # self.dump_paths()
                
    def generate_leaves(self, root):
        # for each python file, the root will always be a module class. 
        fringe = [(root, [])] 
        leaf_nodes = []
        
        while fringe:
          
            node, path = fringe.pop(0)
            trunk = []

            curr_path = path + [node]

            for f in ast.iter_fields(node): 

                if isinstance(f[1], list):
                    for item in f[1]: 
                        if isinstance(item, ast.AST):
                            # print(node, "has_%s"%f[0], item)
                            trunk.append((item, curr_path))

                            if isinstance(item, ast.FunctionDef):
                                print(self.dump_node(item))

                        else:
                            # print(node, "has_%s"%f[0], item)
                            leaf_nodes.append((item, curr_path))
                            # print('leaves:', item, curr_path)
                            # print(item, isinstance(item, ast.AST))
                elif isinstance(f[1], ast.AST):
                    # print(node, "has_%s"%f[0], f[1])
                    # trunk.append(f[1])
                    trunk.append((f[1], curr_path))
                else:
                    if f[1] is not None:
                        # print(node, "has_%s"%f[0], f[1])
                        leaf_nodes.append((f[1], curr_path))
                        # print('leaves:', f[1], curr_path)

            # fringe should be updated to be [trunk;fringe].
            fringe = trunk + fringe
            # fringe = trunk
            
        return leaf_nodes

    # Given a module name, return list of function definitions
    def generate_function_defs(self, root):
        function_defs = []
        for f in ast.iter_fields(root):
            if isinstance(f[1], list):
                for item in f[1]:
                    if isinstance(item, ast.FunctionDef):
                        function_defs.append(item.name)
                        print('Function definition %s found at %s' % (item.name, item.lineno))
        return function_defs

    def generate_paths(self, leaves):
        paths = []
        
        for leaf_pair in itertools.combinations(leaves, 2):
            left_leaf, left_path = leaf_pair[0]
            right_leaf, right_path = leaf_pair[1]
            
            last_common_root_idx = 0 

            # finding the last common root.
            while last_common_root_idx < len(left_path) and last_common_root_idx < len(right_path):
                if left_path[last_common_root_idx] == right_path[last_common_root_idx]:
                    last_common_root_idx += 1
                else:
                    break 

            path = left_path[last_common_root_idx:][::-1] + right_path[last_common_root_idx-1:]
            paths.append((left_leaf, path, right_leaf))
            paths.append((right_leaf, path[::-1], left_leaf))

        return paths

    def dump_node(self, node):
        # utility function.
        print(ast.dump(node))

    def dump_leaf_nodes(self):
        pprint.pprint(self.leaves)

    def dump_paths(self):
        pprint.pprint(self.paths)

    def visualize_ast(self):
        save_path = (Path(self.code_path) / "ast_trees")
        save_path.mkdir(exist_ok=True)
        for py_file in self.ast_nodes:
            file_name = (py_file.split('/')[-1]).split('.')[0] + '.svg'
            module = self.ast_nodes[py_file]
            Settings['terminal_color'] = "#8B0000"
            svg = render(module, Settings)
            with open(str(save_path/file_name), 'w') as f:
                f.write(svg._repr_svg_())

    def export(self, file_name, limit=10):
        # export all the paths contained in self.paths to file_name.txt
        save_path = (Path(self.code_path) / file_name).resolve()
        with open(save_path, 'w') as f:
            for py_file in self.paths:
                for left_leaf, path, right_leaf in self.paths[py_file]:
                    if self.path_length_lower_bound <= len(path) <= self.path_length_upper_bound:
                        path_string = "_".join([type(i).__name__ for i in path])
                        f.write(str(left_leaf)+'\t'+path_string+'\t'+str(right_leaf)+'\n')
                


if __name__ == "__main__":
    explorer = ASTExplorer('../test/fashion_mnist')
    # explorer = ASTExplorer('../test/Alexnet')
    # explorer = ASTExplorer('../test/AGAN')
    explorer.process_all_files()
    explorer.visualize_ast()
    explorer.export("path_triples.txt")
    # import pdb; pdb.set_trace()

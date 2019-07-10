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

import itertools

class ASTExplorer:
    
    def __init__(self, code_path):

        self.code_path = code_path
        self.all_py_files = glob("%s/**/*.py" % str(code_path), recursive=True)
        self.leaves={}
        self.paths=[]
        self.function_defs={}

    def process_all_files(self):

        for py_file in self.all_py_files:
            with open(py_file, "rt", encoding="utf-8") as f:
                content = f.read()

                module_node = ast.parse(content, py_file)

                self.leaves[py_file] = self.generate_leaves(module_node)
                self.function_defs[py_file] = self.generate_function_defs(module_node)
                pdb.set_trace()

                print(py_file, len(self.leaves[py_file]))

                # generate paths
                self.paths += self.generate_paths(self.leaves[py_file])
                # for leaf_pair in itertools.combinations(self.leaves[py_file], 2):
                #     left_leaf, left_path = leaf_pair[0]
                #     right_leaf, right_path = leaf_pair[1]
                    
                #     common_idx = 0 

                #     while common_idx < len(left_path) and common_idx < len(right_path):
                #         if left_path[common_idx] == right_path[common_idx]:
                #             common_idx += 1
                #         else:
                #             break 
                #     # print(common_idx)
                #     # print(left_leaf, left_path)
                #     # print(right_leaf, right_path)
                #     # print(left_leaf, left_path[common_idx:][::-1] + right_path[common_idx-1:], right_leaf)
                #     self.paths.append((left_leaf, left_path[common_idx:][::-1] + right_path[common_idx-1:], right_leaf))
                
                # self.dump_node(module_node)

                
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
            trunk.extend(fringe)
            fringe = trunk
            
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
            
            common_idx = 0 

            while common_idx < len(left_path) and common_idx < len(right_path):
                if left_path[common_idx] == right_path[common_idx]:
                    common_idx += 1
                else:
                    break 
            # print(common_idx)
            # print(left_leaf, left_path)
            # print(right_leaf, right_path)
            # print(left_leaf, left_path[common_idx:][::-1] + right_path[common_idx-1:], right_leaf)
            paths.append((left_leaf, left_path[common_idx:][::-1] + right_path[common_idx-1:], right_leaf))

        return paths

    def dump_node(self, node):
        print(ast.dump(node))

    def dump_leaf_nodes(self):
        print(self.leaves)

    def visualize_ast(self):
        pass # TODO

if __name__ == "__main__":
    # explorer = ASTExplorer('../test/fashion_mnist')
    # explorer = ASTExplorer('../test/Alexnet')
    # explorer = ASTExplorer('../test/AGAN')
    #explorer = ASTExplorer('../test/NPRF-master')
    explorer = ASTExplorer('../test/fashion-mnist')

    explorer.process_all_files()
    import pdb; pdb.set_trace()

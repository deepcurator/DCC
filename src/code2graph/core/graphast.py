import ast, pprint, itertools, astor, re
from glob import glob
from pathlib import Path

from .pyan.analyzer import CallGraphVisitor
from .pyan.node import Flavor

class Doc2vecDataExtractor:

    ''' this class is for creating dataset for doc2vec '''

    def __init__(self, code_path, store_path):
        # path to code repository
        self.code_path = code_path
        self.store_path = store_path

        # path of all python files
        self.all_py_files = glob("%s/**/*.py" % str(code_path), recursive=True)

        # acquire the trunk of extracted ast nodes, all of the nodes are functions.
        self.ast_nodes = self.extract_code_to_astnodes()

    def extract_code_to_astnodes(self):
        ''' generate all the ast nodes according to resolution. '''
        nodes = []

        call_graph_visitor = CallGraphVisitor(self.all_py_files)

        uses_edges = call_graph_visitor.uses_edges
        
        for pyan_node in uses_edges.keys():
            if pyan_node.flavor == Flavor.FUNCTION or pyan_node.flavor == Flavor.METHOD:
                nodes.append((pyan_node.namespace +'.'+ pyan_node.name, pyan_node.ast_node))

        return nodes

    def dump_functions_source_code(self):    
        
        save_path = Path(self.store_path / "doc2vec.txt").resolve()

        with open(str(save_path), 'w') as file:
            
            for name, function_ast in self.ast_nodes:

                # data_helper.xxx.__init__ => ['data', 'helper', 'xxx', 'init']
                keywords = set([x for x in re.split('[^1-9a-zA-Z]', name) if x is not ''])
                splited_function_string = [s.strip() for s in astor.to_source(function_ast).split('\n')]
                
                """ the format of dataset rows """
                file.write('|'.join(keywords) + ' ')
                file.write(' '.join(splited_function_string))
                file.write('\n')


class Code2vecDataExtractor:
    
    def __init__(self, code_path, store_path):
        # path to code repository
        self.code_path = code_path
        self.store_path = store_path

        # path of all python files
        self.all_py_files = glob("%s/**/*.py" % str(code_path), recursive=True)
        
        # leaves nodes of ast tree for each module
        # paths between leaves of ast tree for each module
        self.leaves, self.paths = {}, {} 
        
        # hyperparameters
        self.path_length_upper_bound = 10
        self.path_length_lower_bound = 3
        
        # acquire the trunk of extracted ast nodes, all of the nodes are functions.
        self.ast_nodes = self.extract_code_to_astnodes()
    
    def extract_code_to_astnodes(self):
        ''' generate all the ast nodes according to resolution. '''
        nodes = []

        call_graph_visitor = CallGraphVisitor(self.all_py_files)

        uses_edges = call_graph_visitor.uses_edges
        
        for pyan_node in uses_edges.keys():
            if pyan_node.flavor == Flavor.FUNCTION or pyan_node.flavor == Flavor.METHOD:
                nodes.append((pyan_node.namespace +'.'+ pyan_node.name, pyan_node.ast_node))

        return nodes
        
    def process_all_nodes(self):
        
        ''' acquire leaves with corresponding paths in self.leaves for each starting node (module or function)'''
        ''' acquire paths by iterating through all leave pairs for each starting node (module or function)'''
        self.generate_leaves()
        self.generate_paths()

        for node_name, node in self.ast_nodes:            
            print(node_name, ' gets %s leaves' % len(self.leaves[node_name]))
            print(node_name, ' gets %s paths' %  len(self.paths[node_name]))

    def generate_leaves(self):
        # for each python file, the root will always be a module class. 
        # the node is the root of the AST subtree. (can be module/functiondef)
        # a tuple is used for each item in DFS (ast node and visited paths)

        for node_name, node in self.ast_nodes:

            fringe = [(node, [])] 
            leaf_nodes = []
            
            while fringe:
              
                node, path = fringe.pop(0)
                trunk = []

                curr_path = path + [node]

                for f in ast.iter_fields(node): 

                    if isinstance(f[1], list):
                        for item in f[1]: 
                            if isinstance(item, ast.AST):
                                trunk.append((item, curr_path))

                                if isinstance(item, ast.FunctionDef):
                                    pass
                                    # print(self.dump_node(item))

                            else:
                                leaf_nodes.append((item, curr_path))

                    elif isinstance(f[1], ast.AST):
                        trunk.append((f[1], curr_path))
                    
                    else:
                        if f[1] is not None:
                            leaf_nodes.append((f[1], curr_path))

                # fringe should be updated to be [trunk;fringe].
                fringe = trunk + fringe
                
            self.leaves[node_name] = leaf_nodes

    def generate_paths(self):

        for node_name, node in self.ast_nodes:

            paths = []
            # iterate through leaves in pairs. 
            for leaf_pair in itertools.combinations(self.leaves[node_name], 2):
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

            self.paths[node_name] = paths

    def export(self):
        # export all the paths contained in self.paths to file_name.txt
        # file name contains labels. 
        save_path = (Path(self.store_path) / 'code2vec.txt').resolve()

        with open(str(save_path), 'w') as f:

            for node_name in self.paths:
                keywords = set()
                x = node_name.split('.')
                if x[-1] == "__init__":
                    keywords.add(x[-2])
                else:
                    keywords.add(x[-1])
                # keywords = set([x for x in re.split('[^1-9a-zA-Z]', node_name) if x is not ''])
                f.write('|'.join(keywords) + ' ')
                

                for left, path, right in self.paths[node_name]:
                    
                    if len(path) >= 2 and (type(path[-1]).__name__ == "Str" and type(path[-2]).__name__ == "Expr"):
                        ## skip comment in left leaf for easier data pre-processing. 
                        continue

                    if len(path) >= 2 and (type(path[0]).__name__ == "Str" and type(path[1]).__name__ == "Expr"):
                        ## skip comment in right leaf for easier data pre-processing. 
                        continue
                        
                    if self.path_length_lower_bound <= len(path) <= self.path_length_upper_bound:
                        path_string = "_".join([type(i).__name__ for i in path])
                        f.write("%s\t%s\t%s" % (str(left).replace(' ', ''), path_string, str(right).replace(' ', '')))
                        f.write(' ')
                
                f.write('\n')
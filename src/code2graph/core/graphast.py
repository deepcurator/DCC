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
        
        dirpath = Path(self.store_path / ('functions_source_code')).resolve()
        dirpath.mkdir(exist_ok=True)

        with open(str(dirpath / "doc2vec.txt"), 'w') as file:
            
            for name, function_ast in self.ast_nodes:

                # data_helper.xxx.__init__ => ['data', 'helper', 'xxx', 'init']
                keywords = set([x for x in re.split('[^a-zA-Z]', name) if x is not ''])
                splited_function_string = [s.strip() for s in astor.to_source(function_ast).split('\n')]
                
                """ the format of dataset rows """
                file.write('|'.join(keywords) + ' ')
                file.write(' '.join(splited_function_string))
                file.write('\n')


class Code2vecDataExtractor:
    
    def __init__(self, code_path, resolution="module"):
        # path to code repository
        self.code_path = code_path
        # path of all python files
        self.all_py_files = glob("%s/**/*.py" % str(code_path), recursive=True)
        # leaves nodes of ast tree for each module
        self.leaves={}
        # paths between leaves of ast tree for each module
        self.paths={}

        self.function_defs={}
        
        # hyperparameters
        self.path_length_upper_bound = 10
        self.path_length_lower_bound = 3

        # common parameters
        self.resolution = resolution
        
        # acquire the trunk of extracted ast nodes, all of the nodes are functions.
        self.nodes = self.extract_code_to_astnodes()
    
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
        
        for node_name, node in self.nodes:
            
            print('processing:', node_name)
            
            ''' acquire leaves with corresponding paths in self.leaves for each starting node (module or function)'''
            self.leaves[node_name] = self.generate_leaves(node)

            ''' acquire paths by iterating through all leave pairs for each starting node (module or function)'''
            self.paths[node_name]  = self.generate_paths(self.leaves[node_name])
                        
            print(node_name, ' gets %s leaves' % len(self.leaves[node_name]))
            print(node_name, ' gets %s paths' %  len(self.paths[node_name]))

    def generate_leaves(self, node):
        # for each python file, the root will always be a module class. 

        # the node is the root of the AST subtree. (can be module/functiondef)
        
        # a tuple is used for each item in DFS (ast node and visited paths)

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
                        # print(node, "has_%s"%f[0], f[1])
                        leaf_nodes.append((f[1], curr_path))
                        # print('leaves:', f[1], curr_path)

            # fringe should be updated to be [trunk;fringe].
            fringe = trunk + fringe
            
        return leaf_nodes

    def generate_paths(self, leaves):
        paths = []
        # iterate through leaves in pairs. 
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

    def export(self, stored_path = None):
        # export all the paths contained in self.paths to file_name.txt
        # file name contains labels. 
        if stored_path:
            triple_dir_path = (Path(stored_path) / ('triples_ast_%s'%self.resolution)).resolve()
            triple_dir_path.mkdir(exist_ok=True)
        else:
            triple_dir_path = (Path(self.code_path) / ('triples_ast_%s'%self.resolution)).resolve()
            triple_dir_path.mkdir(exist_ok=True)

        for node_name in self.paths:

            file_name = "%s.txt" % (node_name)
            save_path = triple_dir_path / file_name

            with open(str(save_path), 'w') as f:
                for left_leaf, path, right_leaf in self.paths[node_name]:
                    
                    if len(path) >= 2 and (type(path[-1]).__name__ == "Str" and type(path[-2]).__name__ == "Expr"):
                        ## skip comment in left leaf for easier data pre-processing. 
                        continue

                    if len(path) >= 2 and (type(path[0]).__name__ == "Str" and type(path[1]).__name__ == "Expr"):
                        ## skip comment in right leaf for easier data pre-processing. 
                        continue
                        
                    if self.path_length_lower_bound <= len(path) <= self.path_length_upper_bound:
                        path_string = "_".join([type(i).__name__ for i in path])

                        f.write(str(left_leaf)+'\t'+path_string+'\t'+str(right_leaf)+'\n')
    
    def dump_functions_source_code(self, stored_path = None):
        
        assert self.resolution is "function"
        
        if stored_path:
            dirpath = Path(stored_path / ('functions_source_code')).resolve()
            dirpath.mkdir(exist_ok=True)
        else:
            dirpath = Path(self.code_path / ('functions_source_code')).resolve()
            dirpath.mkdir(exist_ok=True)

        for name, function in self.nodes:
            print(name, function)
            function_string = astor.to_source(function)
            function_name = function_string.split('(')[0].replace('def ','')
            # TODO: Funciton name includes input arguments, 
            # so removing it might remove important information.
            # function_string = ':'.join(function_string.split(':')[1:])
            with open(str(dirpath / (function_name + ".txt")), 'w') as file:
                file.write(function_string.strip().replace('\n', ' <nl>'))
        
        import pdb; pdb.set_trace()

    def dump_node(self, node):
        # utility function.
        print(ast.dump(node))

    def dump_leaf_nodes(self):
        pprint.pprint(self.leaves)

    def dump_paths(self):
        pprint.pprint(self.paths)

if __name__ == "__main__":
    # explorer = ASTExplorer('../test/fashion_mnist')
    # explorer = ASTExplorer('../test/Alexnet')
    # explorer = ASTExplorer('../test/AGAN', resolution="function")
    explorer = ASTExplorer('../test')
    explorer.process_all_nodes()
    # explorer.visualize_ast()
    explorer.export()

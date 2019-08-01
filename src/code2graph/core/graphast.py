import ast, pprint, itertools
from glob import glob
from pathlib import Path


class ASTExplorer:
    
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
        
        # parameters
        self.path_length_upper_bound = 10
        self.path_length_lower_bound = 3
        self.resolution = resolution
        
        # this is the trunk of nodes, the type of it will change depending on resolution. 
        self.nodes = self.get_all_nodes()
    
    def get_all_nodes(self):
        ''' generate all the ast nodes according to resolution. '''
        nodes = {} 

        for py_file in self.all_py_files:
            
            with open(py_file, "rt", encoding="utf-8") as f:
                
                root_node = ast.parse(f.read(), py_file)
                
                module_name = Path(py_file).resolve().stem
                
                if self.resolution is "module":
                    nodes[module_name] = root_node
                
                elif self.resolution is "function":
                    for node in ast.walk(root_node):
                        if isinstance(node, ast.FunctionDef):
                            #TODO need to address __init__ one day. 
                            nodes[module_name +'.'+ node.name] = node

        return nodes
        
    def process_all_nodes(self):
        
        for node_name, node in self.nodes.items():
            
            print('processing:', node_name)
            
            ''' acquire leaves with corresponding paths in self.leaves for each starting node (module or function)'''
            self.leaves[node_name] = self.generate_leaves(node)

            ''' acquire paths by iterating through all leave pairs for each starting node (module or function)'''
            self.paths[node_name]  = self.generate_paths(self.leaves[node_name])
            
            # self.function_defs[node] = self.generate_function_defs(node)
            
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

    # def generate_function_defs(self, root):
    #     # Given a module name, return list of function definitions
    #     function_defs = []
    #     for f in ast.iter_fields(root):
    #         if isinstance(f[1], list):
    #             for item in f[1]:
    #                 if isinstance(item, ast.FunctionDef):
    #                     function_defs.append(item.name)
    #                     print('Function definition %s found at %s' % (item.name, item.lineno))
    #     return function_defs

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

    def visualize_ast(self):
        save_path = (Path(self.code_path) / "ast_trees")
        save_path.mkdir(exist_ok=True)
        # for py_file in self.ast_nodes:
            # file_name = (py_file.split('/')[-1]).split('.')[0] + '.svg'
            # module = self.ast_nodes[py_file]
            # Settings['terminal_color'] = "#8B0000"
            # svg = render(module, Settings)
            # with open(str(save_path/file_name), 'w') as f:
            #     f.write(svg._repr_svg_())

    def export(self):
        # export all the paths contained in self.paths to file_name.txt
        # file name contains labels. 

        triple_dir_path = (Path(self.code_path) / ('triples_ast_%s'%self.resolution)).resolve()
        triple_dir_path.mkdir(exist_ok=True)

        for node_name in self.paths:

            file_name = "%s.txt" % (node_name)
            save_path = triple_dir_path / file_name

            with open(str(save_path), 'w') as f:
                for left_leaf, path, right_leaf in self.paths[node_name]:
                    if self.path_length_lower_bound <= len(path) <= self.path_length_upper_bound:
                        path_string = "_".join([type(i).__name__ for i in path])
                        f.write(str(left_leaf)+'\t'+path_string+'\t'+str(right_leaf)+'\n')
    
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
    explorer = ASTExplorer('../test/AGAN', resolution="function")
    explorer.process_all_nodes()
    # explorer.visualize_ast()
    explorer.export()

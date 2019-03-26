## dataset_manager.py
##
## Utility function to extract the event.Summary file of a given deep learning
## architecture codes.
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
### Defense Advanced Research Projects Agency (DARPA)
### under Agreement No. HR00111990010

import sys, ast, astor, os, pdb
from glob import glob
from pathlib import Path

from matplotlib import colors as mcolors
from pyvis.network import Network
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pprint


class TFcodeInjector:

    def __init__(self, code_repo_path, file_path):

        inject_code = "\nfrom keras import backend as K\nimport tensorflow as tf2\nimport sys\nsess = K.get_session()\nwriter = tf2.summary.FileWriter(\"./\", sess.graph)\nsys.exit()\n"
        self.inject_code_ast_tree = ast.parse(inject_code)

        self.code_repo_path = code_repo_path
        self.code_ast_tree = ast.parse(open(str(code_repo_path / file_path),'r').read().strip())

    def inject_code(self, class_visitor, node):
        
        location = []
        found = False

        for idx, obj in enumerate(node.body):
            visitor = class_visitor()
            visitor.visit(obj)

            if visitor.found:
                if isinstance(obj, (ast.Expr, ast.Call)):
                    location.append(idx)
                elif isinstance(obj, (ast.If, ast.For)):
                    self.inject_code(class_visitor, obj)
                else:
                    print(astor.dump_tree(obj))

        if len(location):
            for idx in location[::-1]:
                node.body.insert(idx+1, self.inject_code_ast_tree)

        
    def inject(self):
        self.inject_code(CompileVisitor, self.code_ast_tree)
        self.inject_code(RunVisitor, self.code_ast_tree)
        f = open(str(self.code_repo_path/'modified.py'), 'w')
        f.write(astor.to_source(self.code_ast_tree))
        f.close()

class CompileVisitor(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "compile":
            self.found = True

class RunVisitor(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "run":
            self.found = True

def tf_code_injection(path):

    for pyfile_path in glob("%s/**/*.py" % str(path), recursive=True):
        injector = TFcodeInjector(path, os.path.basename(pyfile_path))
        injector.inject()


if __name__ == "__main__":    
 
    path = Path("..")/"test"/"fashion_mnist"
    path.resolve()
 
    tf_code_injection(path)
    
    
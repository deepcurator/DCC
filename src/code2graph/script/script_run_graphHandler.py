import sys
sys.path.append('../')

import os
from pathlib import Path

from core.graphHandler import graphHandler
from config.config import GraphHandlerArgParser

def runGraphHandler(path):
    # subdirs = [x for x in path.iterdir() if x.is_dir()]
    
    # for subdir in subdirs:
    #     print(subdir)
    gHandle = graphHandler(logdir=os.path.expanduser(path))

    gHandle.readGraphDef()

    gHandle.convertGraphDef2Json()

    gHandle.writeJson(logdir=str(path))
    gHandle.convertJson2RDF()
    gHandle.saveRDFTriples()
        # gHandle.displayRDF()

if __name__ == "__main__":
    args = GraphHandlerArgParser().get_args()
    path = Path(args.logdir)
    path.resolve()
    runGraphHandler(path)
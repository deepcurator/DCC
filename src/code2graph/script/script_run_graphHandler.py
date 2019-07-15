import sys
sys.path.append('../')

import os
from pathlib import Path

from core.graphHandler import graphHandler
from config.config import GraphHandlerArgParser

def runGraphHandler(path):
    gHandle = graphHandler(logdir=os.path.expanduser(path))

    gHandle.readGraphDef()

    gHandle.convertGraphDef2Json()

    gHandle.writeJson(logdir=str(path))

    gHandle.convertJson2RDF()

    gHandle.displayRDF()

if __name__ == "__main__":
    args = GraphHandlerArgParser().get_args()
    path = Path(args.logdir)
    path.resolve()
    runGraphHandler(path)
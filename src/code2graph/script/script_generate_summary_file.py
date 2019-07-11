import sys
sys.path.append('../')

import os
from glob import glob
from pathlib import Path

from core.automation.code_injection_util import TFcodeInjector
from config.config import GenerateSummaryArgParser

def tf_code_injection(path):

    for pyfile_path in glob("%s/**/*.py" % str(path), recursive=True):
        injector = TFcodeInjector(path, os.path.basename(pyfile_path))
        injector.inject()

if __name__ == "__main__":
    args = GenerateSummaryArgParser().get_args()
    path = Path(args.path)
    path.resolve()
    tf_code_injection(path)
    # run the modifed file with injected code to genearte summary file
    exec(open(str(path/'modified.py')).read(), globals())
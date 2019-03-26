#! /usr/bin/env python2.7
import sys, compileall
result = compileall.compile_dir(sys.argv[1], force=True)
print("Result:", result)